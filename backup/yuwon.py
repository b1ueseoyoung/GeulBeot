import os
import json
import time
from typing import List, Dict, Any, Optional, Set

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.vectorstores import FAISS
from collections import defaultdict

# ==========================
# 전역 상수 / 설정
# ==========================

# 파일/디렉토리 경로
LORE_DB_VECTOR_DIR = "./faiss_lore_db"            # Lore_DB (정규화 설정) 벡터 스토어
FULL_STORY_DB_VECTOR_DIR = "./faiss_full_story_db"  # Full_Story_DB (원본 청크) 벡터 스토어
LORE_DB_FILE = "./lore_db.jsonl"
FULL_STORY_DB_FILE = "./full_story_db.jsonl"
CONFLICT_DB_FILE = "./conflict_db.jsonl"

# 회차 기본값
DEFAULT_EPISODE_SEQ = 1

# 청크 처리 사이 딜레이 (rate limit 대비용, 기본 0: 없음)
PER_CHUNK_DELAY = 0.3

# 벡터 검색 Top-K
SEARCH_TOP_K = 5           # 설정 검색
SEARCH_TOP_K_CONTEXT = 3   # 맥락 검색

# 시스템 프롬프트 캐시
LORE_KEEPER_SYSTEM_PROMPT = ""

# ★ 현재 소설 ID
CURRENT_NOVEL_ID: int = 1

# ==========================
# 글로벌 상태 (in-memory RDB 역할)
# ==========================

current_story_db: List[Dict[str, Any]] = []  # 현재 회차 lore_items
conflict_db: List[Dict[str, Any]] = []       # 충돌 내역
lore_db: List[Dict[str, Any]] = []           # 확정된 lore_items 누적
full_story_db: List[str] = []                # 확정된 원본 청크 누적

_lore_db_vectordb: Optional[FAISS] = None           # Lore_DB 벡터 스토어
_full_story_db_vectordb: Optional[FAISS] = None     # Full_Story_DB 벡터 스토어
_current_story_vectordb: Optional[FAISS] = None     # Current_DB 임시 벡터 스토어
_current_chunk_vectordb: Optional[FAISS] = None      # Current_Chunks 임시 벡터 스토어 (FAISS)

_state_loaded: bool = False

# 현재 처리 중인 회차/청크 인덱스 (충돌 로그용)
CURRENT_EPISODE_SEQ: int = DEFAULT_EPISODE_SEQ
CURRENT_CHUNK_INDEX: int = -1

# 청크마다 검색 결과를 모아두는 컨텍스트
SEARCH_CONTEXT_BY_CHUNK = defaultdict(
    lambda: {
        "lore": "",
        "current": "",
        "full": "",
        "chunks": ""
    }
)

# ==========================
# OpenAI / LLM 클라이언트 (재사용)
# ==========================

# OpenAI API Key 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = OpenAIEmbeddings()

LLM_CLASSIFIER = ChatOpenAI(model="gpt-4o-mini", temperature=0)     # 청크 타입 분류용
LLM_FACT_EXTRACTOR = ChatOpenAI(model="gpt-4o", temperature=0)      # 사실 추출용
LLM_AGENT_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)    # ReAct 에이전트용
CONFLICT_JUDGE_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=5,) # 설정 충돌 판단용

# ==========================
# DB 스키마 Enum 유틸
# ==========================

ITEM_TYPES = {"FACT", "RULE", "EXCEPTION"}
CATEGORIES = {
    "PHY_STATUS", "PHY_TRAIT", "ABILITY", "ITEM",
    "RELATION", "LOCATION", "WORLD_SETTING", "EMOTION"
}
TARGET_GROUPS = {"GLOBAL", "RACE", "CLASS", "INDIVIDUAL"}
CHUNK_TYPES = {"TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D"}


def _normalize_enum(value: Optional[str], allowed: Set[str], default: str) -> str:
    if not value:
        return default
    value = value.upper().strip()
    return value if value in allowed else default


def _build_lore_item(
    raw_fact: Dict[str, Any],
    chunk: str,
    source_seq: int = DEFAULT_EPISODE_SEQ,
    novel_id: Optional[int] = None
) -> Dict[str, Any]:
    """lore_items 스키마에 맞춰 raw fact를 정규화"""

    nid = (
        novel_id
        if novel_id is not None
        else raw_fact.get("novel_id", CURRENT_NOVEL_ID)
    )

    return {
        "novel_id": nid,
        "item_type": _normalize_enum(raw_fact.get("item_type"), ITEM_TYPES, "FACT"),
        "category": _normalize_enum(raw_fact.get("category"), CATEGORIES, "WORLD_SETTING"),
        "target_group": _normalize_enum(raw_fact.get("target_group"), TARGET_GROUPS, "INDIVIDUAL"),
        "chunk_type": _normalize_enum(raw_fact.get("chunk_type"), CHUNK_TYPES, "TYPE_A"),
        "subject": raw_fact.get("subject") or raw_fact.get("actor") or "UNKNOWN",
        "condition": raw_fact.get("condition") or "",
        "effect": raw_fact.get("effect") or raw_fact.get("predicate") or "",
        "text": raw_fact.get("text") or chunk,
        "source_seq": raw_fact.get("source_seq") or source_seq,
        "metadata": raw_fact.get("metadata", {}),
    }


# ==========================
# 벡터 스토어 관리 (Lore_DB / Full_Story_DB)
# ==========================

def get_lore_db_vectordb() -> Optional[FAISS]:
    """Lore_DB 전용 벡터 스토어 (정규화된 lore_items)"""
    global _lore_db_vectordb

    if _lore_db_vectordb is not None:
        return _lore_db_vectordb

    if os.path.exists(LORE_DB_VECTOR_DIR):
        print(f"[Lore_DB_VectorStore] 기존 벡터 스토어 로드: {LORE_DB_VECTOR_DIR}")
        _lore_db_vectordb = FAISS.load_local(
            LORE_DB_VECTOR_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return _lore_db_vectordb

    # 아직 생성된 적이 없음 → None 유지
    return None


def get_full_story_db_vectordb() -> Optional[FAISS]:
    """Full_Story_DB 전용 벡터 스토어 (원본 chunk_data)"""
    global _full_story_db_vectordb

    if _full_story_db_vectordb is not None:
        return _full_story_db_vectordb

    if os.path.exists(FULL_STORY_DB_VECTOR_DIR):
        print(f"[Full_Story_DB_VectorStore] 기존 벡터 스토어 로드: {FULL_STORY_DB_VECTOR_DIR}")
        _full_story_db_vectordb = FAISS.load_local(
            FULL_STORY_DB_VECTOR_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return _full_story_db_vectordb

    return None


def add_to_lore_db_vectorstore(lore_items: List[Dict[str, Any]]):
    """Current_DB의 정규화된 lore_items를 Lore_DB 벡터 스토어에 추가"""
    if not lore_items:
        return

    global _lore_db_vectordb

    texts = []
    metadatas = []

    for item in lore_items:
        searchable_text = (
            f"[{item['item_type']}/{item['category']}] "
            f"Subject: {item['subject']}, "
            f"Effect: {item['effect']}, "
            f"Condition: {item['condition']}, "
            f"Text: {item['text']}"
        )
        texts.append(searchable_text)
        metadatas.append({
            "novel_id": item.get("novel_id", CURRENT_NOVEL_ID),
            "item_type": item["item_type"],
            "category": item["category"],
            "target_group": item["target_group"],
            "chunk_type": item["chunk_type"],
            "subject": item["subject"],
            "condition": item["condition"],
            "effect": item["effect"],
            "text": item["text"],
            "source_seq": str(item["source_seq"]),
        })

    # 아직 로드/생성된 인덱스가 없다면 → 새로 생성
    if _lore_db_vectordb is None:
        print("[Lore_DB_VectorStore] 새 FAISS 인덱스 생성 및 초기 데이터 적재")
        _lore_db_vectordb = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
    else:
        print("[Lore_DB_VectorStore] 기존 인덱스에 데이터 추가")
        _lore_db_vectordb.add_texts(texts=texts, metadatas=metadatas)

    # 디스크에 항상 저장
    _lore_db_vectordb.save_local(LORE_DB_VECTOR_DIR)
    print(f"[Lore_DB_VectorStore] {len(texts)}개의 lore_items가 벡터화되어 저장되었습니다.")


def add_to_full_story_db_vectorstore(
        chunk_data_list: List[str],
        episode_seq: int,
        chunk_types: Optional[List[str]] = None,
        novel_id: Optional[int] = None
    ):
    """원본 chunk_data를 Full_Story_DB 벡터 스토어에 추가"""
    if not chunk_data_list:
        return

    global _full_story_db_vectordb

    texts = chunk_data_list
    metadatas = []
    for i, _ in enumerate(chunk_data_list):
        ct = (chunk_types[i] if chunk_types and i < len(chunk_types) else "TYPE_D")
        meta = {
            "novel_id": novel_id,
            "chunk_index": i,
            "source": "processed_episode",
            "source_seq": episode_seq,              # ★ 추가
            "chunk_type": ct,                        # ★ 추가
        }
        metadatas.append(meta)

    if _full_story_db_vectordb is None:
        print("[Full_Story_DB_VectorStore] 새 FAISS 인덱스 생성 및 초기 데이터 적재")
        _full_story_db_vectordb = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
    else:
        print("[Full_Story_DB_VectorStore] 기존 인덱스에 데이터 추가")
        _full_story_db_vectordb.add_texts(texts=texts, metadatas=metadatas)

    _full_story_db_vectordb.save_local(FULL_STORY_DB_VECTOR_DIR)
    print(f"[Full_Story_DB_VectorStore] {len(chunk_data_list)}개의 chunk_data가 벡터화되어 저장되었습니다.")


# ==========================
# Current_Chunks (FAISS, 회차 단위 임시 벡터 스토어)
# ==========================

def get_current_chunk_vectordb() -> Optional[FAISS]:
    """현재 회차 원문 청크용 임시 벡터 스토어(FAISS)를 반환합니다."""
    return _current_chunk_vectordb


def add_to_current_chunk_vectorstore(chunk: str, episode_seq: int = CURRENT_EPISODE_SEQ):
    """
    지금 청크 하나를 Current_Chunks 벡터스토어에 적재.
    """
    if not chunk:
        return

    global _current_chunk_vectordb, CURRENT_CHUNK_INDEX

    texts = [chunk]
    metadatas = [{
        "novel_id": CURRENT_NOVEL_ID,
        "chunk_index": CURRENT_CHUNK_INDEX, 
        "source_seq": str(episode_seq),
    }]

    if _current_chunk_vectordb is None:
        print(f"[Current_Chunks] 새 인덱스 생성, 총 1개 적재")
        _current_chunk_vectordb = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
    else:
        print(f"[Current_Chunks] 기존 인덱스에 1개 추가 (chunk_index={CURRENT_CHUNK_INDEX})")
        _current_chunk_vectordb.add_texts(texts=texts, metadatas=metadatas)


# ==========================
# Current_DB (현재 회차 정규화 설정용 벡터 스토어)
# ==========================

def get_current_story_vectordb() -> Optional[FAISS]:
    """Current_DB 전용 벡터 스토어 (회차 단위 임시, 정규화한 데이터, non-persist)"""
    return _current_story_vectordb


def add_to_current_db_vectorstore(lore_items: List[Dict[str, Any]]):
    """Current_DB에 저장된 lore_items를 임시 벡터 스토어에 추가"""
    if not lore_items:
        return

    global _current_story_vectordb

    texts = []
    metadatas = []

    for item in lore_items:
        searchable_text = (
            f"[{item['item_type']}/{item['category']}] "
            f"Subject: {item['subject']}, "
            f"Effect: {item['effect']}, "
            f"Condition: {item['condition']}, "
            f"Text: {item['text']}"
        )
        texts.append(searchable_text)
        metadatas.append({
            "novel_id": item.get("novel_id", CURRENT_NOVEL_ID),
            "item_type": item["item_type"],
            "category": item["category"],
            "target_group": item["target_group"],
            "chunk_type": item["chunk_type"],
            "subject": item["subject"],
            "condition": item["condition"],
            "effect": item["effect"],
            "text": item["text"],
            "source_seq": str(item["source_seq"]),
        })

    if _current_story_vectordb is None:
        print(f"[Current_DB_VectorStore] 새 인덱스 생성, {len(texts)}개 적재")
        _current_story_vectordb = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
    else:
        print(f"[Current_DB_VectorStore] 기존 인덱스에 {len(texts)}개 추가")
        _current_story_vectordb.add_texts(texts=texts, metadatas=metadatas)


# ==========================
# 상태 관리 / 파일 I/O
# ==========================

def reset_current_episode_state():
    """현재 회차 상태 초기화 (current/ conflict/ 임시 벡터)"""
    current_story_db.clear()
    conflict_db.clear()
    global _current_story_vectordb, _current_chunk_vectordb
    _current_story_vectordb = None
    _current_chunk_vectordb = None


def _load_dataframe(path: str) -> Optional[pd.DataFrame]:
    """json/jsonl 형식의 파일을 Dataframe으로 가져오기."""
    if not os.path.exists(path):
        return None
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jsonl", ".json"}:
            return pd.read_json(path, lines=(ext == ".jsonl"))
        return pd.read_csv(path)
    except Exception:
        return None


def _write_dataframe(df: pd.DataFrame, path: str):
    """DataFrame으로 JSON/JSONL 형식으로 저장."""
    is_jsonl = os.path.splitext(path)[1].lower() == ".jsonl"
    df.to_json(
        path,
        orient="records",
        force_ascii=False,
        lines=is_jsonl,
        indent=None if is_jsonl else 2,
    )


def append_conflicts_to_file(conflicts: List[Dict[str, Any]], path: str):
    """conflict_db 내용을 파일에 누적 저장 (JSON/JSONL)"""
    if not conflicts:
        return
    df_new = pd.DataFrame(conflicts)
    df_old = _load_dataframe(path)
    if df_old is not None:
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    _write_dataframe(df_new, path)
    print(f"[Conflict_Log] {len(conflicts)}건 기록 → {path}")


def save_lore_db_to_file(items: List[Dict[str, Any]], path: str = LORE_DB_FILE):
    if not items:
        return
    rows = []
    for it in items:
        row = dict(it)
        row["metadata"] = json.dumps(row.get("metadata", {}), ensure_ascii=False)
        rows.append(row)
    df_new = pd.DataFrame(rows)
    df_old = _load_dataframe(path)
    if df_old is not None:
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    _write_dataframe(df_new, path)
    print(f"[Lore_DB_File] {len(items)}개 저장/누적 → {path}")


def save_full_story_to_file(chunks: List[str], path: str = FULL_STORY_DB_FILE, episode_seq: int = DEFAULT_EPISODE_SEQ, chunk_types: Optional[List[str]] = None, novel_id: Optional[int] = None):
    if not chunks:
        return

    nid = novel_id if novel_id is not None else CURRENT_NOVEL_ID

    rows = []
    for i, ck in enumerate(chunks):
        ct = (chunk_types[i] if chunk_types and i < len(chunk_types) else "TYPE_D")
        row = {
            "novel_id": nid,
            "text": ck,
            "source_seq": episode_seq,
            "chunk_index": i,
            "chunk_type": ct,     # ★ 추가
        }
        rows.append(row)
    df_new = pd.DataFrame(rows)
    df_old = _load_dataframe(path)
    if df_old is not None:
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    _write_dataframe(df_new, path)
    print(f"[Full_Story_File] {len(chunks)}개 저장/누적 → {path}")


def load_persistent_state():
    """
    json/jsonl에서 lore_db/full_story_db를 로드하고,
    벡터 디렉토리가 없을 때만 초기 임베딩을 수행.
    """
    global _state_loaded
    if _state_loaded:
        return

    lore_vectordb_exists = os.path.exists(LORE_DB_VECTOR_DIR)
    full_vectordb_exists = os.path.exists(FULL_STORY_DB_VECTOR_DIR)

    # Lore_DB
    if os.path.exists(LORE_DB_FILE):
        try:
            df = _load_dataframe(LORE_DB_FILE)
            if df is not None:
                for _, row in df.iterrows():
                    metadata = {}
                    if isinstance(row.get("metadata"), str):
                        try:
                            metadata = json.loads(row["metadata"])
                        except Exception:
                            metadata = {}
                    elif isinstance(row.get("metadata"), dict):
                        metadata = row.get("metadata") or {}

                    lore_db.append(
                        {
                            "novel_id": row.get("novel_id", 0),
                            "item_type": row.get("item_type", "FACT"),
                            "category": row.get("category", "WORLD_SETTING"),
                            "target_group": row.get("target_group", "INDIVIDUAL"),
                            "chunk_type": row.get("chunk_type", "TYPE_A"),
                            "subject": row.get("subject", "UNKNOWN"),
                            "condition": row.get("condition", ""),
                            "effect": row.get("effect", ""),
                            "text": row.get("text", ""),
                            "source_seq": row.get("source_seq", DEFAULT_EPISODE_SEQ),
                            "metadata": metadata,
                        }
                    )
            if lore_db and not lore_vectordb_exists:
                add_to_lore_db_vectorstore(lore_db)
                print(f"[Load] lore_db {len(lore_db)}개 로드 및 벡터 반영 (새로 생성)")
            else:
                print(f"[Load] lore_db {len(lore_db)}개 로드 (기존 벡터스토어 사용)")
        except Exception as e:
            print(f"[Load] lore_db 로드 실패: {e}")

    # Full_Story_DB
    if os.path.exists(FULL_STORY_DB_FILE):
        try:
            df = _load_dataframe(FULL_STORY_DB_FILE)
            if df is not None:
                texts = df["text"].tolist()
                full_story_db.extend(texts)
                if texts and not full_vectordb_exists:
                    # novel_id / source_seq / chunk_index까지 메타로 넘겨주기
                    records = df.to_dict("records")  # 각 row가 dict
                    meta_list = []
                    for i, r in enumerate(records):
                        meta_list.append(
                            {
                                "novel_id": r.get("novel_id", 0),
                                "source_seq": r.get("source_seq", DEFAULT_EPISODE_SEQ),
                                "chunk_index": r.get("chunk_index", i),
                                "source": "load",
                            }
                        )
                    _full_story_db_vectordb.save_local(FULL_STORY_DB_VECTOR_DIR)
                    print(f"[Load] full_story_db {len(texts)}개 로드 및 벡터 반영 (새로 생성)")
                else:
                    print(f"[Load] full_story_db {len(texts)}개 로드 (기존 벡터스토어 사용)")
        except Exception as e:
            print(f"[Load] full_story_db 로드 실패: {e}")

    _state_loaded = True


# ==========================
# 에이전트 수행 결과 처리
# ==========================

def save_to_current_db(chunk: str, facts_json: str) -> str:
    """
    [Current_DB 저장]
    충돌이 없는 설정을 Current_DB에 저장합니다.
    """
    try:
        facts = json.loads(facts_json) if facts_json else []

        # 이미 있는 FACT 키셋 가져오기
        existing_keys = {
            (item["item_type"], item["category"], item["subject"], item["effect"])
            for item in current_story_db
        }

        normalized = []
        for fact in facts:
            key = (
                fact.get("item_type"),
                fact.get("category"),
                fact.get("subject"),
                fact.get("effect"),
            )
            # 완전히 같은 설정은 한 번만 저장
            if key in existing_keys:
                continue

            item = _build_lore_item(
                fact,
                chunk,
                source_seq=CURRENT_EPISODE_SEQ,
                novel_id=CURRENT_NOVEL_ID,
            )
            normalized.append(item)
            existing_keys.add(key)

        current_story_db.extend(normalized)
        add_to_current_db_vectorstore(normalized)
        return f"✓ Current_DB에 {len(normalized)}개 저장됨 (중복/추측 제외)"
    except Exception as e:
        return f"✗ 저장 실패: {str(e)}"


def report_conflict_to_db(
    chunk: str,
    conflict_type: str,
    reason: str,
    facts_json: str = "[]",
    conflicting_text: str = "",
) -> str:
    """
    [충돌 리포트]
    감지된 설정 충돌을 Conflict_DB에 저장합니다.

    사용 규칙:
    - conflict_type: "Hard Conflict" 또는 "Soft Conflict" 포함
    - reason: 아래 정보가 명확히 드러나도록 기술
      * 현재 문장의 어떤 설정이
      * 어떤 기존 설정(어느 DB: Lore_DB / Current_DB / Full_Story_DB, subject/effect/source_seq 등)과
      * 왜 충돌하는지 (시간, 상태, 규칙 위반 등)
    - conflicting_text: 실제로 충돌하는 기존 설정 또는 원문 일부를 그대로 붙여넣기
    """
    try:
        facts = json.loads(facts_json) if facts_json else []
        normalized_type = "Hard Conflict" if "hard" in conflict_type.lower() else "Soft Conflict"

        entry = {
            "source_seq": CURRENT_EPISODE_SEQ,
            "chunk_index": CURRENT_CHUNK_INDEX,
            "input_text": chunk,
            "is_conflict": True,
            "conflict_type": normalized_type,
            "evidence": reason,
            "facts": facts,
            "conflicting_text": conflicting_text,
        }
        conflict_db.append(entry)

        msg = f"✗ {normalized_type} 감지됨"
        if conflicting_text:
            msg += f"\n  기존 설정 일부: {conflicting_text[:50]}..."
        return msg
    except Exception as e:
        return f"✗ 저장 실패: {str(e)}"

def _extract_from_messages(messages):
    facts_json = "[]"
    judge_result = None
    search_context = "[]"

    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = msg.name or ""

            if name == "extract_facts_from_chunk":
                facts_json = msg.content.strip()

            elif name == "judge_conflict":
                raw = msg.content
                judge_result = json.loads(raw)

            elif name in [
                "search_lore_db",
                "search_current_db",
                "search_full_story_db",
                "search_current_chunks",
            ]:
                ctx = SEARCH_CONTEXT_BY_CHUNK.get(CURRENT_CHUNK_INDEX, {})
                search_context = "\n\n".join(
                    filter(
                        None,
                        [
                            ctx.get("lore", ""),
                            ctx.get("current", ""),
                            ctx.get("full", ""),
                            ctx.get("chunks", ""),
                        ],
                    )
                )

    return facts_json, judge_result, search_context



# ==========================
# Tools 정의 (LangChain Tools)
# ==========================

@tool
def search_current_db(query: str) -> str:
    """
    [Current_DB 검색 도구]
    현재 회차에서 정규화된 설정을 임시 벡터 스토어에서 검색합니다.
    """
    if not current_story_db:
        return "[Current_DB가 비어있습니다]"

    vectordb = get_current_story_vectordb()
    if vectordb is None:
        return "[Current_DB 벡터스토어가 아직 생성되지 않았습니다]"
    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K, filter={"novel_id": CURRENT_NOVEL_ID})

    if not docs:
        return "[Current_DB에서 관련 설정 없음]"

    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        lines.append(
            f"{i}. [Current_DB] [{meta.get('item_type')}/{meta.get('category')}] "
            f"{meta.get('subject')} -> {meta.get('effect')}\n"
            f"   [source_seq={meta.get('source_seq', '?')}] 원문: {meta.get('text', '')}"
        )

    result = "[Current_DB 검색 결과 (임시 설정)]\n" + "\n\n".join(lines)
    SEARCH_CONTEXT_BY_CHUNK[CURRENT_CHUNK_INDEX]["current"] = result
    return result


@tool
def search_lore_db(query: str) -> str:
    """
    [Lore_DB 검색 도구]
    이전 회차들에서 확정된 설정들(정규화된 lore_items)을 벡터 검색합니다.
    """
    if not lore_db:
        return "[Lore_DB가 비어있습니다]"

    vectordb = get_lore_db_vectordb()
    if vectordb is None:
        return "[Lore_DB 벡터스토어가 아직 생성되지 않았습니다]"
    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K, filter={"novel_id": CURRENT_NOVEL_ID})

    if not docs:
        return "[Lore_DB에서 관련 설정 없음]"

    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        lines.append(
            f"{i}. [Lore_DB] [{meta.get('item_type')}/{meta.get('category')}] "
            f"{meta.get('subject')} -> {meta.get('effect')}\n"
            f"   [source_seq={meta.get('source_seq', '?')}] 원문: {meta.get('text', '')}"
        )

    result = "[Lore_DB 검색 결과 (정규화된 설정)]\n" + "\n\n".join(lines)
    SEARCH_CONTEXT_BY_CHUNK[CURRENT_CHUNK_INDEX]["lore"] = result
    return result


@tool
def search_full_story_db(query: str) -> str:
    """
    [Full_Story_DB 검색 도구]
    이전 회차들의 원본 chunk_data를 벡터 검색합니다 (맥락 확인용).
    """
    if not full_story_db:
        return "[Full_Story_DB가 비어있습니다]"

    vectordb = get_full_story_db_vectordb()
    if vectordb is None:
        return "[Full_Story_DB 벡터스토어가 아직 생성되지 않았습니다]"
    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K_CONTEXT, filter={"novel_id": CURRENT_NOVEL_ID})

    if not docs:
        return "[Full_Story_DB에서 관련 맥락 없음]"

    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        lines.append(
            f"[Full_Story_DB] chunk_index={meta.get('chunk_index', '?')}, source={meta.get('source', '?')}\n"
            f"{i}. {doc.page_content}"
        )
    result = "[Full_Story_DB 검색 결과 (원본 맥락)]\n" + "\n\n".join(lines)
    SEARCH_CONTEXT_BY_CHUNK[CURRENT_CHUNK_INDEX]["full"] = result
    return result


@tool
def search_current_chunks(query: str) -> str:
    """
    [Current_Chunks 검색 도구]
    현재 회차의 원문 청크를 임시 벡터 스토어에서 검색합니다.
    """
    vectordb = get_current_chunk_vectordb()
    if vectordb is None:
        return "[Current_Chunks가 비어있습니다]"

    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K_CONTEXT, filter={"novel_id": CURRENT_NOVEL_ID})

    if not docs:
        return "[Current_Chunks에서 관련 맥락 없음]"

    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        lines.append(
            f"{i}. [Current_Chunks] (chunk_index={meta.get('chunk_index', '?')}, "
            f"source_seq={meta.get('source_seq', '?')})\n"
            f"   {doc.page_content}"
        )
    result = "[Current_Chunks 검색 결과 (원문)]\n" + "\n\n".join(lines)
    SEARCH_CONTEXT_BY_CHUNK[CURRENT_CHUNK_INDEX]["chunks"] = result
    return result


# @tool
# def get_current_db_settings() -> str:
#     """
#     [Current_DB 조회 도구]
#     현재 회차에서 확정된 설정들을 조회합니다.
#     """
#     if not current_story_db:
#         return "[현재 회차에서 확정된 설정이 아직 없습니다]"

#     lines = []
#     for item in current_story_db[-20:]:  # 최근 20개만
#         lines.append(
#             f"- [{item['item_type']}/{item['category']}] "
#             f"{item['subject']} -> {item['effect']}\n"
#             f"  [source_seq={item.get('source_seq', '?')}] 원문: {item['text']}"
#         )

#     return "[Current_DB 설정들]\n" + "\n".join(lines)


@tool
def classify_chunk_type(chunk: str) -> str:
    """
    [청크 타입 분류 도구]
    A(설정), B(감정), C(대화), D(단순 서술)로 분류합니다.
    """
    llm = LLM_CLASSIFIER

    system_msg = (
        "너는 웹소설 청크 분류 AI야.\n"
        "주어진 문장을 A/B/C/D로 분류해:\n"
        "A: 사실/설정 (세계관, 인물/아이템 상태, 사건, 배경, 공간, 시간, 계절, 날씨, 이후 이야기에 영향을 주는 요소)\n"
        "B: 감정/내면 (심리, 내면 독백)\n"
        "C: 대화 (직접 화법)\n"
        "D: 단순 서술 (정보 없는 묘사, 의성어, 욕설, 비유 등)\n"
        "반드시 A/B/C/D 한 글자만 출력해."
    )

    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"분류:\n{chunk}"}
    ])

    text = resp.content.strip().upper()
    for ch in ["A", "B", "C", "D"]:
        if ch in text:
            return ch
    return "D"


@tool
def extract_facts_from_chunk(chunk: str, chunk_type: str) -> str:
    """
    [설정 추출 도구]
    lore_items 스키마를 따르는 JSON 배열을 추출합니다.
    TYPE_A/B/C로 분류된 청크에 대해서만 호출.
    """
    llm = LLM_FACT_EXTRACTOR

    system_msg = (
        "너는 웹소설 설정 추출 AI야.\n"
        "주어진 문장에서 lore_items 스키마를 추출해:\n"
        "item_type: FACT/RULE/EXCEPTION\n"
        "category: PHY_STATUS/PHY_TRAIT/ABILITY/ITEM/RELATION/LOCATION/EMOTION/EVENT/TIME/WEATHER/GUESS/WORLD_SETTING\n"
        "target_group: GLOBAL/RACE/CLASS/INDIVIDUAL\n"
        "chunk_type: TYPE_A/TYPE_B/TYPE_C\n"
        f"chunk_type 필드는 반드시 '{chunk_type}'로 넣어.\n"
        "subject, condition, effect, text를 포함한 JSON 배열만 출력.\n"
        "\n"
        "TYPE_A(설정/세계관): 인물 상태/직업/관계/경제상태와, 시대/장소/날씨/사회 분위기, 플롯에 의미 있는 사건 같은 배경·세계관 정보를 가능한 한 세분화해서 FACT/RULE/EXCEPTION으로 추출해.\n"
        "- **한 번 일어난 중요한 사건(예: A가 B를 죽였다, 결혼했다, 전쟁이 났다, 해고되었다 등)도 설정으로 추출해.\n" 
        "- 이런 건 category를 EVENT로 두고, subject는 행위 주체, effect에는 간단한 사건 내용을 쓴다.**\n"
        "TYPE_B(감정/내면): 앞으로 전개나 관계에 영향을 줄 만큼 강한 '지속적인 감정 상태'나 '결심'을 우선 설정으로 추출해.\n"
        "                 예: 왕에 대한 불신, 동료를 지키겠다는 결심 등. category는 EMOTION 또는 PHY_STATUS.\n"
        "- 다만, 이 기준에 애매하게 걸리는 경우라도, 현재 장면에서 명확하게 드러나는 강한 감정(크게 기쁨, 깊은 슬픔, 극도의 분노 등)이 있다면 그것도 EMOTION FACT로 추출.\n"
        "TYPE_C(대화): 약속·맹세·관계 선언·규칙 언급·장기 목표처럼 나중에 다시 참조될 내용만 FACT/RULE로 추출해.\n"
        "              단순 잡담/의성어/짧은 감탄사는 추출하지 마.\n"
        "\n"
        "[사실 추출시 강력 규칙]\n"
        "1. 모든 요청에 대해 반드시 추출을 해야 한다. []와 같은 빈 배열 출력 불가.\n"
        "   정말 애매하면, 문장의 핵심 내용을 한 줄로 요약한 설정을 만들어.\n"
        "2. 인물에 대한 신상 정보(이름, 특히 나이, 소속, 직업, 가족, 건강 등)는 무조건 추출한다.\n"
        "   - 특히, 중심 인물과 관련된 인물 정보는 사소한 것도 전부 추출한다."
        "3. 주요한 행동, 사건에 대한 정보는 무조건 추출한다.\n"
        "4. '~할 수도 있다', '~일 것이다', '~인 듯하다'처럼 모호한 표현, 개인의 추측은 effect에서 추측이라는 것을 명시한다.\n"
        "   - 이 경우 category를 GUESS로 분류한다."
        "   - 같은 chunk 안에서 같은 내용에 대해 스스로 부정하거나 단정짓는 표현이 나오면, 단정하는 부분만 설정으로 추출해라."
        "   (예: '아니다', '결코 아니다'처럼 스스로 부정하는 표현 -> '가기 싫어서가 아니다'처럼 단정하는 쪽만 추출출)\n"
        "5. 인물 대사나 욕설 속 과장된 표현, 반어법은 설정으로 추출하지 않는다.\n"
        "   (예: '조밥도 못 먹는 년' → 욕설이지 건강 상태 설정이이 아님)\n"
        "6. 대화를 설정으로로 저장할 때는, 반드시 대화문임을 명확히 인지하고, 화자와 청자의 상태를 확인해야 한다.\n"
        "   - 화자와 청자는 명확하게 인물의 이름(그, 그녀와 같은 인칭 대명사 불가)을 명시한다. 만약 이름을 찾을 수 없으면 사실로 추출하지 않는다.\n"
        "7. 대화문에서 화자가 술에 취해 있거나 거짓말을 하는 경우, 그 내용은 설정이 아니다.\n"
        "   - 단, 청자에게는 FACT일 수 있다. 그 경우에는 들은 정보임을 명시하라.(예: 그가 사람을 죽였다는 것을 K가 들었다 ->  K는 그가 사람을 죽였다고 들음)\n"
        "8. subject가 인물일 경우, 무조건 대상의 이름(김첨지), 이름이 없을 경우 주요 인물과의 관계(김첨지의 아내)로 표현한다.\n"
        "   - 그, 그녀, 그 애와 같은 인칭 대명사로 표현하지 않는다.\n"
        "   - 화자, 청자와 같은 대명사로 표현하지 않는다.\n"
        "   - 정확한 대상을 알 수 없다면 맥락에 따라 추론하되, 추론의 결과가 확실하지 않다면 사실로 추출하지 않는다.\n"
        "   - subject가 무조건 인물인 것은 아니다. 이 설정이 무엇/누구에 대한 설명인지에 대한 정보값, 즉 서술의 대상이다."
        "9. 설정으로 추출한 내용은 반드시 원문에 명확히 드러나 있어야 한다.\n"
        "10. 절대 원문에 없는 새로운 설정을 상상해서 FACT를 생성하지 않는다.\n"
        "\n"
        "[item_type 의미]\n"
        "- FACT: 지금 장면에서 실제로 벌어진 상태/사건 (시간·공간·상태가 구체적).\n"
        "- RULE: 세계/사회/마법/직업 상의 일반적인 규칙, 인물의 가치관 등 개인의 신조/습관/미신 등 믿음.\n"
        "- EXCEPTION: 기존 RULE을 깨는 '이 경우만 예외' 설정.\n"
    )


    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"문장:\n{chunk}"}
    ])

    raw = resp.content.strip()
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        return raw[start:end]
    except Exception:
        return "[]"


@tool
def judge_conflict(chunk: str, facts_json: str) -> str:
    """
    [충돌 판정 도구]
    - chunk: 현재 청크 원문
    - facts_json: extract_facts_from_chunk로 추출된 FACT 리스트 (JSON 문자열)

    규칙:
    - 오직 "Hard Conflict"만 판정한다.
    - Soft Conflict는 사용하지 않는다 (conflict_type은 "" 또는 "Hard Conflict"만).
    - EvidenceContext 안에 있는 문장과 FACT가 "논리적으로 동시에 참일 수 없을 때"만 is_conflict=true.
    - 조금이라도 애매하면 is_conflict=false.
    """
    llm = CONFLICT_JUDGE_LLM

    ctx = SEARCH_CONTEXT_BY_CHUNK.get(CURRENT_CHUNK_INDEX, {})
    evidence_context = "\n\n".join(
        filter(
            None,
            [
                ctx.get("lore", ""),
                ctx.get("current", ""),
                ctx.get("full", ""),
                ctx.get("chunks", ""),
            ],
        )
    )
    # print(evidence_context)

    conflict_guidelines = """
[Hard Conflict 정의]

Hard Conflict는 "A AND (NOT A)" 타입의 **논리적으로 양립 불가능한 사실**만 의미한다.
조금이라도 해석/추론/감정이 섞이면 Hard Conflict가 아니다.

다음과 같은 경우에만 Hard Conflict로 인정한다.

1) 생사/존재 모순
   - 이전: '아내는 이미 죽었다'
   - 현재: '아내가 걸어왔다 / 말을 했다 / 밥을 먹었다'
   -> 동시에 참일 수 없음.

2) 시간/날짜 순서 모순 (동일 사건에 대해)
   - 이전: '이 사건은 3월 5일에 일어났다'
   - 현재: '같은 사건이 3월 1일에 일어났다'
   -> 같은 사건이 같은 세계선에서 다른 날짜에 일어날 수 없음.

3) 위치 모순 (동일 인물/대상 + 같은 시점)
   - 이전: '김첨지는 그 시간에 서울에 있었다'
   - 현재: '같은 시간에 부산에서 김첨지가 술을 마셨다'
   -> 한 사람이 같은 시간에 두 곳에 동시에 있을 수 없음.

4) 수치/개수/돈 등 명시적 수량 모순
   - 이전: '오늘 한 푼도 벌지 못했다'
   - 현재: '오늘 30원을 벌었다'
   - 이전: '아이를 전혀 낳지 못했다'
   - 현재: '그의 아들은 ...'
   -> 0 vs 1 이상처럼, 한쪽이 확실한 '없음'인데 다른 쪽이 '있음'인 경우.

5) 특성/능력의 논리적 모순
   - 이전: 'A는 말을 할 수 없다(벙어리다)'
   - 현재: 'A가 길게 연설했다'
   - 이전: '이 캐릭터는 불사신이다'
   - 현재: '완전히 죽어서 다시는 돌아올 수 없다'
   -> 정의 자체와 정면으로 반대.

[절대 Hard Conflict가 아닌 것들 (전부 PASS)]

- 감정 변화, 이상한 행동, 슬픈데 웃는 것, 기쁜데 우는 것
- '이해 안 되는 행동', '개연성이 부족한 것처럼 느껴지는 행동'
- 비유, 은유, 상징, 과장된 표현, 욕설
- '운수 좋은 날인데 실제로는 비극적인 결말' 같은 아이러니
- 인물의 심리 상태, 분위기, 감정 표현
- LLM이 "성격상 그럴 리 없다"라고 상상해서 만든 것
- 과거 회상 등 시간/날짜가 현재가 아닌 경우

[판단 절차]

1) FACTS_JSON과 EvidenceContext에서, "같은 대상/속성"에 대해
   서로 정반대 내용을 말하는 문장 쌍이 있는지 찾는다.
   - '있다 vs 없다', '살아있다 vs 죽었다', '0 vs 1 이상', '서울 vs 부산(같은 시점)' 등.

2) 그런 쌍을 **명확하게** 찾지 못했다면:
   - is_conflict = false
   - conflict_type = ""
   - reason = "명시적인 Hard Conflict 없음 (모든 차이는 상태 변화/새 정보/해석 차이로 볼 수 있음)"

3) 그런 쌍을 하나라도 찾으면:
   - is_conflict = true
   - conflict_type = "Hard Conflict"
   - reason에 "어떤 문장과 어떤 문장이 왜 동시에 참일 수 없는지"를 한국어로 요약.
   - conflicting_text에는 EvidenceContext 안에 실제로 있는, '기존 설정 문장' 하나를 그대로 넣는다.

**[최종 리포트 가이드]**
- Hard Conflict일 경우: 반드시 DB에서 충돌하는 문장(Evidence)을 찾아 `conflicting_text`에 넣어라.
- Soft Conflict일 경우: "기존의 [어떤 말/행동]으로 보아 [어떤 성격]인데, 현재 행동은 이에 위배된다"고 해석해라.
- 반드시 search_lore_db / search_current_db / search_full_story_db / search_current_chunks
  도구의 **출력 문자열 안에 실제로 존재하는 문장만** `conflicting_text`로 사용할 수 있다.
  네가 머릿속으로 상상한 문장이나, 도구 출력에 없는 내용은 Evidence로 쓰면 안 된다.
  예를 들어, DB 어디에도 '의사에게 보인 적이 있다'라는 표현이 없으면,
  그런 문장을 근거로 충돌을 만들지 마라.

**충돌이 아닌 경우(다음 상황에서는 절대 충돌로 판단하지 마):**
- 작품에서 처음 등장하는 인물/아이템/장소/사건 등이 소개되는 경우.
- 기존 내용을 부정하지 않고, 새로운 설명이나 세부 정보를 덧붙이는 경우
- 같은 인물/대상이 '시간이 흐르면서' 상태가 변화한 것처럼 읽히는 경우
- 같은 내용을 다른 말로 반복하거나 요약하는 단순 재서술.
- 두 사실이 모두 동시에 참일 수 있고, A이면서 동시에 'not A'가 되지 않는 경우
- 일반적인 성장·심경 변화처럼 서사의 흐름상 자연스러운 변화.
- 판단이 모호하거나 애매하거나 억지로 해석해야만 충돌처럼 보이는 경우
- 추측이나 가능성이 명백한 FACT와 다른 경우 (FACT에 대하여 충돌 판단하지 않는다.)
- 비유나 개인의 가치관을 소개하는 경우
- 실제로 행동한 것이 아니라 단순히 서술상에서 '~했다면', '~하면'과 같이 가정을 하는 경우
        """

    system_msg = (
        "너는 '설정 충돌 전용 판정기'야.\n"
        "오직 Hard Conflict(논리적/물리적으로 동시에 참일 수 없는 경우)만 판단한다.\n"
        "EvidenceContext, FACTS_JSON, CHUNK에 **존재하는 문장만** 근거로 사용해야 한다.\n"
        "네 지식/상식/추론으로 새로운 설정을 만들면 안 된다.\n"
        "애매하다 싶으면 무조건 is_conflict=false로 하라.\n"
        + conflict_guidelines
    )

    user_msg = (
        f"[CHUNK]\n{chunk}\n\n"
        f"[FACTS_JSON]\n{facts_json}\n\n"
        f"[EVIDENCE_CONTEXT]\n{evidence_context}\n"
        "위 정보를 바탕으로 설정 충돌 여부를 판정하고, 반드시 아래래 JSON 형식으로만 답해:\n"
        "{"
        "\"is_conflict\": true/false, "
        "\"conflict_type\": \"Hard Conflict\" 또는 \"\", "
        "\"reason\": \"...\", "
        "\"conflicting_text\": \"...\""
        "}"
    )

    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])

    raw = resp.content.strip()
    
    # 안전장치: conflicting_text가 EvidenceContext 안에 없으면 강제로 충돌 아님으로 변경
    try:
        data = json.loads(raw)
    except Exception:
        return json.dumps(
            {
                "is_conflict": False,
                "conflict_type": "",
                "reason": "judge_conflict 응답 파싱 실패, 충돌 없음으로 처리.",
                "conflicting_text": "",
                "source_seq": ""
            },
            ensure_ascii=False,
        )

    conflicting_text = (data.get("conflicting_text") or "").strip()
    if conflicting_text and conflicting_text not in evidence_context:
        data["is_conflict"] = False
        data["conflict_type"] = ""
        data["reason"] = (
            "conflicting_text가 EvidenceContext 안에 존재하지 않아 "
            "환각으로 간주, 충돌 없음으로 처리."
        )
        data["conflicting_text"] = ""
        data["source_seq"] = ""

    # Soft Conflict는 사용하지 않으므로, 다른 값이 있어도 강제로 정리
    if data.get("conflict_type") not in ("", "Hard Conflict"):
        data["conflict_type"] = "Hard Conflict" if data.get("is_conflict") else ""

    return json.dumps(data, ensure_ascii=False)



# ==========================
# Lore Keeper 에이전트 생성
# ==========================

def create_lore_keeper_agent(model_name: str = "gpt-4o-mini"):
    """Lore Keeper 에이전트 생성"""
    tools = [
        search_lore_db,
        search_current_db,
        search_current_chunks,
        # get_current_db_settings,
        classify_chunk_type,
        extract_facts_from_chunk,
        # save_to_current_db,
        # report_conflict_to_db,
        search_full_story_db,
        judge_conflict
    ]

    # 기본은 재사용 LLM
    llm = LLM_AGENT_MODEL if model_name == "gpt-4o-mini" else ChatOpenAI(model=model_name, temperature=0)

    if CURRENT_EPISODE_SEQ == 1:
        workflow = (
            "1. classify_chunk_type으로 A/B/C/D 분류\n"
            "2. D 타입이면 PASS (작업 종료)\n"
            "3. A/B/C 타입이면:\n"
            "   a. extract_facts_from_chunk로 사실 추출\n"
            # "   b. 이후 save_to_current_db\n"
        )
    else:
        workflow = (
            "1. classify_chunk_type으로 A/B/C/D 분류\n"
            "2. D 타입이면 PASS (작업 종료)\n"
            "3. A/B/C 타입이면:\n"
            "   a. extract_facts_from_chunk로 사실 추출\n"
            "   c. search_lore_db (이전 회차 정규화된 설정 검색)\n"
            "   d. search_current_db (현재 회차 임시 설정 검색)\n"
            "   e. search_current_chunks (현재 회차 원문 맥락 검색)\n"
            "   f. search_full_story_db (이전 회차 원문 맥락 검색)\n"
            # "   f. get_current_db_settings (현재 회차 설정 조회)\n"
            "4. judge_conflict 도구를 호출해 추출한 사실과 기존 설정 비교 (충돌 검사 - 필수!)\n"
            # "   h. judge_conflict 도구의 판정 결과에 따라: \n"
            # "    * is_conflict가 true이면 반드시시 report_conflict_to_db 호출 \n"
            # "    * is_conflict가 false이면 반드시 save_to_current_db 호출\n"
            "*c, d를 통해 찾은 설정에 현재 문장에서 검증해야하는 내용이 완벽히 들어있지 않을 때 e, f를 수행한다.\n"
            "   - 이때, 충분하지 않다는 것은 찾은 설정들이 현재의 문장과 관련이 없거나 동떨어진 정보일 경우, 판단을 하기에 정보가 부족한 경우를 말한다.\n"
            "   (예 : 개똥이는 다섯 살이다 -> 개똥이의 나이에 대한 정보가 찾은 설정 중에 없어서 search_current_chunks, search_full_story_db 호출.)\n"
        )
        

    system_prompt = f"""너는 웹소설 설정 관리 AI 에이전트 'Lore Keeper'야.

**작업 절차:**
{workflow}

**최초 회차(이전 Lore_DB / Full_Story_DB가 비어 있을 때)의 원칙:**
- 이 회차에서 나오는 설정은 '기준 설정'이므로 어떤 내용도 충돌로 판단하지 않는다.
- 이 경우 search_* 도구를 호출하지 않고 종료

**최종 응답 형식(반드시 이 형식을 그대로 따를 것):**
- 네가 모든 tool 호출과 판단을 마친 뒤, 사용자에게 보내는 마지막 답변은
  아래 2줄 형식을 정확히 지켜서 출력해야 한다. 다른 형태로 말하지 마.

**chunk_type 사용 규칙 (tool 인자용)**  ← (위치 분리)
- 먼저 classify_chunk_type 도구로 청크를 A/B/C/D 중 하나로 분류해.
- 분류 결과를 다음과 같이 TYPE_* 형식으로 매핑해서 사용해:
  - A → "TYPE_A"
  - B → "TYPE_B"
  - C → "TYPE_C"
  - D → "TYPE_D"

1) "충돌 여부: 충돌"  또는  "충돌 여부: 충돌 없음"
2) "이후 동작: <한 줄 요약>"
   - 예시:
     - "이후 동작: Current_DB에 설정 저장 (추후 Lore_DB 반영 후보)"
     - "이후 동작: Conflict_DB에만 기록, Lore_DB / Full_Story_DB 반영 보류"
     - "이후 동작: TYPE_D 서술로 판단하여 저장/충돌 없이 PASS"

- 위 2줄 외에 불필요한 문장, 마크다운, 리스트, 설명은 출력하지 마.

지금부터 주어진 문장을 분석해서 설정 충돌을 검사해줘."""
    global LORE_KEEPER_SYSTEM_PROMPT
    LORE_KEEPER_SYSTEM_PROMPT = system_prompt

    agent_graph = create_react_agent(
        model=llm,
        tools=tools,
    )
    return agent_graph


# ==========================
# 청크 단위 처리
# ==========================

def process_chunk_with_agent(agent_graph, chunk: str, index: int) -> Dict[str, Any]:
    """에이전트로 청크 하나 처리 + current 데이터 저장/충돌 리포트 작성"""
    global CURRENT_CHUNK_INDEX, CURRENT_EPISODE_SEQ
    CURRENT_CHUNK_INDEX = index  # 이 청크에서 발생하는 충돌은 chunk_index로 기록

    print(f"\n{'='*60}")
    print(f"[Chunk {index+1}] {chunk[:60]}...")
    print(f"{'='*60}")

    input_message = HumanMessage(content=f"다음 문장을 분석해서 설정 충돌을 검사해줘:\n\n{chunk}")
    messages_payload = []
    if LORE_KEEPER_SYSTEM_PROMPT:
        messages_payload.append(SystemMessage(content=LORE_KEEPER_SYSTEM_PROMPT))
    messages_payload.append(input_message)

    try:

        result = agent_graph.invoke(
            {"messages": messages_payload},
            {"recursion_limit": 20}
        )

        messages = result["messages"]
        output = messages[-1].content

        print(f"\n[최종 판정] {output[:150]}...")

        # FACT, judge_conflict, 검색 컨텍스트 뽑기
        facts_json, judge_result, search_context = _extract_from_messages(messages)
        print("\n[DEBUG] messages 종류:")
        for m in messages:
            print("  -", type(m), getattr(m, "name", None))

        is_conflict = False
        conflict_type = "None"
        conflicting_text = ""

        # judge_conflict 결과 기반 분기
        if judge_result and judge_result.get("is_conflict"):
            # 추가 가드: conflicting_text가 실제 검색 컨텍스트에 없으면 환각으로 간주
            conflicting_text_candidate = (judge_result.get("conflicting_text") or "").strip()
            if conflicting_text_candidate and conflicting_text_candidate not in search_context:
                # Evidence가 실제 RAG 결과에 없으면 → 충돌로 인정 X
                print("[guard] conflicting_text가 검색 결과에 없어서 충돌 취소 처리.")
                save_to_current_db(chunk, facts_json)
                add_to_current_chunk_vectorstore(chunk, CURRENT_EPISODE_SEQ)
            else:
                is_conflict = True
                conflict_type = judge_result.get("conflict_type", "Unknown")
                conflicting_text = conflicting_text_candidate

                # 💥 실제로 DB에 충돌 기록
                msg = report_conflict_to_db(
                    chunk=chunk,
                    conflict_type=conflict_type,
                    reason=judge_result.get("reason", ""),
                    facts_json=facts_json,
                    conflicting_text=conflicting_text,
                )
                print(msg)
        else:
            # ✅ 충돌 없음 → 무조건 저장
            msg = save_to_current_db(chunk, facts_json)
            add_to_current_chunk_vectorstore(chunk, CURRENT_EPISODE_SEQ)
            print(msg)

#        chunk_type은 facts_json에서 추론, 없으면 TYPE_D
        chunk_type = "TYPE_D"
        try:
            facts = json.loads(facts_json)
            if facts:
                chunk_type = facts[0].get("chunk_type", "TYPE_A")
        except Exception:
            pass

        return {
            "chunk_type": chunk_type,
            "is_conflict": is_conflict,
            "conflict_type": conflict_type,
            "conflicting_text": conflicting_text,
            "agent_output": output,
        }

    except Exception as e:
        print(f"[에러] {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "chunk_type": "Error",
            "is_conflict": False,
            "conflict_type": "None",
            "error": str(e),
        }


# ==========================
# 회차 단위 실행 (main 엔트리)
# ==========================

def run_manual_episode(
    chunks: List[str],
    episode_seq: int = 1,
    novel_id: int = 0,
    clear_after: bool = False,
    conflict_log_path: Optional[str] = None,
    lore_db_path: str = LORE_DB_FILE,
    full_story_path: str = FULL_STORY_DB_FILE,
):
    """
    ground_truth 없이, 사용자가 텍스트 청크 리스트를 넣어 한 회차를 처음부터 끝까지 처리.
    - current/conflict 임시 상태를 초기화하고 시작
    - 충돌 없으면 lore_db/full_story_db 및 벡터 스토어에 반영
    """
    global CURRENT_EPISODE_SEQ, CURRENT_NOVEL_ID
    CURRENT_EPISODE_SEQ = episode_seq
    CURRENT_NOVEL_ID = novel_id

    load_persistent_state()
    reset_current_episode_state()

    print("\n" + "="*60)
    print(f"수동 회차 처리 시작 (episode_seq={episode_seq}, chunks={len(chunks)})")
    print("="*60 + "\n")

    # 벡터 스토어 준비 (lore/full 검색용)
    get_lore_db_vectordb()
    get_full_story_db_vectordb()

    # 현재 회차 원문 청크 → FAISS에 한 번에 적재
    # add_to_current_chunk_vectorstore(chunks, episode_seq=episode_seq)

    agent = create_lore_keeper_agent()
    results = []

    for i, chunk in enumerate(chunks):
        res = process_chunk_with_agent(agent, chunk, index=i)
        results.append(res)
        if PER_CHUNK_DELAY > 0:
            time.sleep(PER_CHUNK_DELAY)

    # 각 청크의 chunk_type 리스트 추출
    chunk_types = [r.get("chunk_type", "TYPE_D") for r in results]

    # 최종 요약 및 후속 동작
    if conflict_db:
        print(f"\n✗ 충돌 {len(conflict_db)}건 감지 - Lore_DB / Full_Story_DB 반영 보류")
        summary = {
            "status": "conflict",
            "source_seq": episode_seq,
            "chunk_count": len(chunks),
            "conflict_count": len(conflict_db),
            "action": "Lore_DB / Full_Story_DB 반영 보류 (Conflict_DB에만 저장)",
        }
        log_path = conflict_log_path or CONFLICT_DB_FILE
        append_conflicts_to_file(conflict_db, log_path)
    else:
        print("\n✓ 충돌 없음 - Lore_DB / Full_Story_DB에 반영")
        lore_db.extend(current_story_db)
        save_lore_db_to_file(current_story_db, lore_db_path)
        add_to_lore_db_vectorstore(current_story_db)

        full_story_db.extend(chunks)
        add_to_full_story_db_vectorstore(chunks, episode_seq=episode_seq, chunk_types=chunk_types, novel_id=CURRENT_NOVEL_ID,)
        save_full_story_to_file(chunks, full_story_path, episode_seq=episode_seq, chunk_types=chunk_types, novel_id=CURRENT_NOVEL_ID)

        summary = {
            "status": "ok",
            "source_seq": episode_seq,
            "chunk_count": len(chunks),
            "conflict_count": 0,
            "saved_lore_items": len(current_story_db),
            "action": "Lore_DB / Full_Story_DB 및 벡터 스토어에 저장",
        }

        

    # 콘솔용 결과 요약
    print("\n=== Lore Keeper 결과 요약 ===")
    print(f"- 회차: {summary.get('source_seq')}")
    print(f"- 청크 수: {summary.get('chunk_count')}")
    conflict_label = "충돌" if summary["status"] == "conflict" else "충돌 없음"
    print(f"- 충돌 여부: {conflict_label} (총 {summary.get('conflict_count', 0)}건)")
    print(f"- 이후 동작: {summary.get('action')}")
    if summary["status"] != "conflict":
        print(f"- 저장된 설정 개수(Current_DB → Lore_DB): {summary.get('saved_lore_items', 0)}")

    # 충돌 상세
    if conflict_db:
        print("\n▶ 충돌 상세")
        for idx, c in enumerate(conflict_db, start=1):
            epi = c.get("source_seq", "?")
            ch_idx = c.get("chunk_index", "?")
            print(f"  [{idx}] 회차 {epi}, 청크 {ch_idx}번")
            print(f"      유형: {c.get('conflict_type')}")
            print(f"      사유: {c.get('reason')}")
            if c.get("conflicting_text"):
                snippet = c["conflicting_text"].replace("\n", " ")
                if len(snippet) > 100:
                    snippet = snippet[:100] + "..."
                print(f"      충돌 대상 원문: {snippet}")

    if clear_after:
        reset_current_episode_state()

    return summary, results


def run_multiple_episodes(
    all_episode_chunks: List[List[str]],
    novel_id: int = 0,
    conflict_log_path: str = CONFLICT_DB_FILE,   # ← 통합 로그 파일 (예: "./conflict_db.jsonl")
    lore_db_path: str = LORE_DB_FILE,
    full_story_path: str = FULL_STORY_DB_FILE,
):
    """
    여러 회차(예: 10회차)를 한 번에 순차 처리하는 헬퍼 함수.

    all_episode_chunks: 각 회차별 청크 리스트의 리스트
        예) [chunks_ep1, chunks_ep2, ..., chunks_ep10]
    """
    summaries = []
    all_results = []

    for ep_idx, chunks in enumerate(all_episode_chunks, start=1):
        print("\n" + "#" * 70)
        print(f"[Multi Runner] {ep_idx}회차 처리 시작 (청크 수={len(chunks)})")
        print("#" * 70 + "\n")

        summary, results = run_manual_episode(
            chunks=chunks,
            episode_seq=ep_idx,
            novel_id=novel_id,
            clear_after=False,
            conflict_log_path=conflict_log_path,  # ← 매 회차 동일 파일에 append
            lore_db_path=lore_db_path,
            full_story_path=full_story_path,
        )
        summaries.append(summary)
        all_results.append(results)

    return summaries, all_results

# ==========================
# 예시 main (직접 실행 시)
# ==========================

if __name__ == "__main__":
    # 예시: 10회차 텍스트를 미리 쪼개서 준비했다고 가정
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separators=[
            r"(?<=[.?!])\s+",
            "\n",
            "\n\n",
        ],
        is_separator_regex=True,
        length_function=len,
    )

    # 여기에 실제 1~10회차 원문들을 넣으면 됨
    texts = [
        """새침하게 흐린 품이 눈이 올 듯하더니 눈은 아니 오고 얼다가 만 비가 추적추적 내리는 날이었다. 이날이야말로 동소문 안에서 인력거꾼 노릇을 하는 김첨지에게는 오래간만에도 닥친 운수 좋은 날이었다. 문안에(거기도 문밖은 아니지만) 들어간답시는 앞집 마마님을 전찻길까지 모셔다 드린 것을 비롯으로 행여나 손님이있을까 하고 정류장에서 어정어정하며 내리는 사람 하나하나에게 거의 비는 듯한 눈결을 보내고 있다가 마침내 교원인 듯한 양복쟁이를 동광학교까지 태워다 주기로 되었다. 첫 번에 삼십전 , 둘째 번에 오십전 - 아침 댓바람에 그리 흉치 않은 일이었다. 그야말로 재수가 옴붙어서 근 열흘 안 돈 구경도 못한 김첨지는 십전짜리 백동화 서 푼, 또는 다섯 푼이 찰깍 하고 손바닥에 떨어질 제 거의 눈물을 흘릴 만큼 기뻤었다. 더구나 이날 이때에 이 팔십 전이라는 돈이 그에게 얼마나 유용한지 몰랐다. 컬컬한 목에 모주 한 잔도 적실 수 있거니와 그보다도 앓는 아내에게 설렁탕 한 그릇도 사다 줄 수 있음이다.""",
        """그의 아내가 기침으로 쿨룩거리기는 벌써 달포가 넘었다. 조밥도 굶기를 먹다시피 하는 형편이니 물론 약 한 첩 써본 일이 없다. 구태여 쓰려면 못 쓸 바도 아니로되 그는 병이란 놈에게 약을 주어 보내면 재미를 붙여서 자꾸 온다는 자기의 신조에 어디까지 충실하였다. 따라서 의사에게 보인 적이 없으니 무슨 병인지는 알 수 없으되 반듯이 누워 가지고 일어나기는 새로 모로도 못 눕는 걸 보면 중증은 중증인 듯. 병이 이대도록 심해지기는 열흘전에 조밥을 먹고 체한 때문이다. 그때도 김첨지가 오래간만에 돈을 얻어서 좁쌀 한 되와 십 전짜리 나무 한 단을 사다 주었더니 김첨지의 말에 의지하면 그 오라질 년이 천방지축으로 냄비에 대고 끓였다. 마음은 급하고 불길은 달지 않아 채 익지도 않은 것을 그 오라질년이 숟가락은 고만두고 손으로 움켜서 두 뺨에 주먹덩이 같은 혹이 불거지도록 누가 빼앗을듯이 처박질하더니만 그날 저녁부터 가슴이 땡긴다, 배가 켕긴다고 눈을 흡뜨고 지랄병을 하였다. 그때 김첨지는 열화와 같이 성을 내며, “에이, 오라질년, 조랑복은 할 수가 없어, 못 먹어 병, 먹어서 병! 어쩌란 말이야! 왜 눈을 바루 뜨지 못해!” 하고 앓는 이의 뺨을 한 번 후려갈겼다. 그는 평생 단 한 번도 아내를 손찌검한 적이 없는 선량한 남편이었다. 흡뜬 눈은 조금 바루어졌건만 이슬이 맺히었다. 김첨지의 눈시울도 뜨끈뜨끈하였다. 이 환자가 그러고도 먹는 데는 물리지 않았다. 사흘 전부터 설렁탕 국물이 마시고 싶다고 남편을 졸랐다. “이런 오라질 년! 조밥도 못 먹는 년이 설렁탕은. 또 처먹고 지랄병을 하게.” 라고, 야단을 쳐보았건만, 못 사주는 마음이 시원치는 않았다.""",
        """인제 설렁탕을 사줄 수도 있다. 앓는 어미 곁에서 배고파 보채는 개똥이(세살먹이)에게 죽을 사줄 수도 있다 - 팔십 전을 손에 쥔 김 첨지의 마음은 푼푼하였다. 그러나 그의 행운은 그걸로 그치지 않았다. 땀과 빗물이 섞여 흐르는 목덜미를 기름주머니가 다된 왜목 수건으로 닦으며, 그 학교 문을 돌아 나올 때 였다. 뒤에서 “인력거!” 하고 부르는 소리가 난다. 자기를 불러 멈춘 사람이 그 학교 학생인 줄 김첨지는 한번 보고 짐작할 수 있었다. 그 학생은 다짜고짜로, “남대문 정거장까지 얼마요.”라고 물었다. 아마도 그 학교 기숙사에 있는 이로 동기방학을 이용하여 귀향하려 함이리라. 오늘 가기로 작정은 하였건만 비는 오고, 짐은 있고 해서 어찌할 줄 모르다가 마침 김첨지를 보고 뛰어나왔음이리라. 그렇지 않으면 왜 구두를 채 신지 못해서 질질 끌고, 비록 고구라 양복일망정 노박이로 비를 맞으며 김첨지를 뒤쫓아 나왔으랴. “남대문 정거장까지 말씀입니까.” 하고 김첨지는 잠깐 주저하였다. 그는 이 우중에 우장도 없이 그 먼 곳을 철벅거리고 가기가 싫었음일까? 처음 것 둘째 것으로 고만 만족하였음일까? 아니다 결코 아니다. 이상하게도 꼬리를 맞물고 덤비는 이 행운 앞에 조금 겁이 났음이다. 그리고 집을 나올 제 아내의 부탁이 마음이 켕기었다 - 앞집 마마님한테서 부르러 왔을 제 병인은 뼈만 남은 얼굴에 유일의 샘물 같은 유달리 크고 움푹한 눈에 애걸하는 빛을 띄우며, “오늘은 나가지 말아요. 제발 덕분에 집에 붙어 있어요. 내가 이렇게 아픈데…….” 라고, 모기 소리같이 중얼거리고 숨을 걸그렁걸그렁하였다. 그때에 김첨지는 대수롭지 않은듯이, “아따, 젠장맞을 년, 별 빌어먹을 소리를 다 하네. 맞붙들고 앉았으면 누가 먹여 살릴 줄 알아.” 하고 훌쩍 뛰어나오려니까 환자는 붙잡을 듯이 팔을 내저으며, “나가지 말라도 그래, 그러면 일찍이 들어와요.” 하고, 목메인 소리가 뒤를 따랐다. 정거장까지 가잔 말을 들은 순간에 경련적으로 떠는 손 유달리 큼직한 눈울 듯한 아내의 얼굴이 김첨지의 눈앞에 어른어른하였다. 아내는 지금까지 한번도 아프지 않았다. 아주 멀쩡했다.""",
        """“그래 남대문 정거장까지 얼마란 말이요?” 하고 학생은 초조한 듯이 인력거꾼의 얼굴을 바라보며 혼자말같이, “인천 차가 열한 점에 있고 그 다음에는 새로 두 점이든가.”라고 중얼거린다. “일 원 오십 전만 줍시요.” 이 말이 저도 모를 사이에 불쑥 김첨지의 입에서 떨어졌다. 제 입으로 부르고도 스스로 그 엄청난 돈 액수에 놀랐다. 한꺼번에 이런 금액을 불러라도 본 지가 그 얼마 만인가! 그러자 그 돈벌 용기가 병자에 대한 염려를 사르고 말았다. 설마 오늘 내로 어떠랴 싶었다. 무슨 일이 있더라도 제일 제이의 행운을 곱친 것보다고 오히려 갑절이 많은 이 행운을 놓칠 수 없다 하였다. “일 원 오십 전은 너무 과한데.” 이런 말을 하며 학생은 고개를 기웃하였다. “아니올시다. 잇수로 치면 여기서 거기가 시오 리가 넘는답니다. 또 이런 진날은 좀 더 주셔야지요.” 하고 빙글빙글 웃는 차부의 얼굴에는 숨길 수 없는 기쁨이 넘쳐흘렀다. “그러면 달라는 대로 줄 터이니 빨리 가요.” 관대한 어린 손님은 이런 말을 남기고 총총히 옷도 입고 짐도 챙기러 갈 데로 갔다. 그 학생을 태우고 나선 김첨지의 다리는 이상하게 거뿐하였다. 달음질을 한다느니보다 거의 나는 듯하였다. 바퀴도 어떻게 속히 도는지 구른다느니 보다 마치 얼음을 지쳐 나가는 스케이트 모양으로 미끄러져 가는 듯하였다. 언 땅에 비가 내려 미끄럽기도 하였지만. 이윽고 끄는 이의 다리는 무거워졌다. 자기 집 가까이 다다른 까닭이다. 새삼스러운 염려가 그의 가슴을 눌렀다. “오늘은 나가지 말아요, 내가 이 렇게 아픈데” 이런 말이 잉잉 그의 에 울렸다. 그리고 병자의 움쑥 들어간 눈이 원망하는 듯이 자기를 노리는 듯하였다. 그러자 엉엉 하고 우는 개똥이의 곡성을 들은 듯싶다. 딸국딸국 하고 숨 모으는 소리도 나는 듯싶다. “왜 이리우, 기차 놓치겠구먼.” 하고 탄 이의 초조한 부르짖음이 간신히 그의 귀에 들어왔다. 언뜻 깨달으니 김첨지는 인력거를 쥔 채 길 한복판에 엉거주춤 멈춰 있지 않은가. “예, 예.” 하고, 김첨지는 또다시 달음질하였다. 집이 차차 멀어 갈수록 김첨지의 걸음에는 다시금 신이 나기 시작하였다. 다리를 재게 놀려야만 쉴새없이 자기의 머리에 떠오르는 모든 근심과 걱정을 잊을 듯이. 정거장까지 끌어다 주고 그 깜짝 놀란 일 원 오십 전을 정말 제 손에 쥠에 제 말마따나 십리나 되는 길을 비를 맞아 가며 질퍽거리고 온 생각은 아니하고 거저나 얻은 듯이 고마웠다. 졸부나 된 듯이 기뻤다. 제 자식뻘밖에 안 되는 어린 손님에게 몇 번 허리를 굽히며, “안녕히 다녀옵시요.” 라고 깍듯이 재우쳤다.""",
        """그러나 빈 인력거를 털털거리며 이 우중에 돌아갈 일이 꿈밖이었다. 노동으로 하여 흐른 땀이 식어지자 굶주린 창자에서, 물 흐르는 옷에서 어슬어슬 한기가 솟아나기 비롯하매 일 원 오십 전이란 돈이 얼마나 괜찮고 괴로운 것인 줄 절절히 느끼었다. 정거장을 떠나는 그의 발길은 힘 하나 없었다. 온몸이 옹송그려지며 당장 그 자리에 엎어져 못 일어날 것 같았다. “젠장맞을 것, 이 비를 맞으며 빈 인력거를 털털거리고 돌아를 간담. 이런 빌어먹을 제 할미를 붙을 비가 왜 남의 상판을 딱딱 때려!” 그는 몹시 화증을 내며 누구에게 반항이나 하는 듯이 게걸거렸다. 그럴 즈음에 그의 머리엔 또 새로운 광명이 비쳤나니 그것은  ‘이러구 갈 게 아니라 이 근처를 빙빙 돌며 차 오기를 기다리면 또 손님을 태우게 될는지도 몰라’란 생각이었다. 오늘 운수가 괴상하게도 좋으니까 그런 요행이 또 한번 없으리라고 누가 보증하랴. 꼬리를 굴리는 행운이 꼭 자기를 기다리고 있다고 내기를 해도 좋을 만한 믿음을 얻게 되었다. 그렇다고 정거장 인력거꾼의 등쌀이 무서우니 정거장 앞에 섰을 수는 없었다. 그래 그는 이전에도 여러 번 해본 일이라 바로 정거장 앞 전차 정류장에서 조금 떨어지게 사람 다니는 길과 전찻길 틈에 인력거를 세워 놓고 자기는 그 근처를 빙빙 돌며 형세를 관망하기로 하였다. 얼마 만에 기차는 왔고 수십 명이나 되는 손이 정류장으로 쏟아져 나왔다. 그 중에서 손님을 물색하는 김첨지의 눈엔 양머리에 뒤축 높은 구두를 신고 망토까지 두른 기생 퇴물인 듯 난봉 여학생인 듯한 여편네의 모양이 띄었다. 그는 슬근슬근 그 여자의 곁으로 다가들었다. “아씨, 인력거 아니 타시랍시요.” 그 여학생인지 만지가 한참은 매우 때깔을 빼며 입술을 꼭 다문 채 김첨지를 거들떠보지도 않았다. 김첨지는 구걸하는 거지나 무엇같이 연해연방 그의 기색을 살피며, “아씨, 정거장 애들보담 아주 싸게 모셔다 드리겠습니다. 댁이 어디신가요.” 하고 추근추근하게도 그 여자의 들고 있는 일본식 버들고리짝에 제 손을 대었다. “왜 이래, 남 귀치않게.” 소리를 벽력같이 지르고는 돌아선다. 김첨지는 어랍시요 하고 물러섰다.""",
        """전차는 왔다. 김첨지는 원망스럽게 전차 타는 이를 노리고 있었다. 그러나 그의 예감은 틀리지 않았다. 전차가 빡빡하게 사람을 싣고 움직이기 시작하였을 제 타고 남은 손 하나가 있었다. 굉장하게 큰 가방을 들고 있는 걸 보면 아마 붐비는 차 안에 짐이 크다 하여 차장에게 밀려 내려온 눈치였다. 김첨지는 대어섰다. “인력거를 타시랍시요.” 한동안 값으로 승강이를 하다가 육십 전에 인사동까지 태워다 주기로 하였다. 인력거가 무거워지매 그의 몸은 이상하게도 가벼워졌고 그리고 또 인력거가 가벼워지니 몸은 다시금 무거워졌건만 이번에는 마음조차 초조해 온다. 집의 광경이 자꾸 눈앞에 어른거리어 인제 요행을 바랄 여유도 없었다. 나무 등걸이나 무엇 같고 제 것 같지도 않은 다리를 연해 꾸짖으며 질팡갈팡 뛰는 수밖에 없었다. 저놈의 인력거꾼이 저렇게 술이 취해 가지고 이 진땅에 어찌 가노, 라고 길 가는 사람이 걱정을 하리만큼 그의 걸음은 황급하였다. 흐리고 비 오는 하늘은 어둠침침하게 벌써 황혼에 가까운 듯하다. 창경원 앞까지 다다라서야 그는 턱에 닿은 숨을 돌리고 걸음도 늦추잡았다. 한 걸음 두 걸음 집이 가까워 갈수록 그의 마음조차 괴상하게 누그러웠다. 그런데 이 누그러움은 안심에서 오는 게 아니요 자기를 덮친 무서운 불행을 빈틈없이 알게 될 때가 박두한 것을 두리는 마음에서 오는 것이다. 그는 불행에 다닥치기 전 시간을 얼마쯤이라도 늘이려고 버르적거렸다. 기적에 가까운 벌이를 하였다는 기쁨을 할 수 있으면 오래 지니고 싶었다. 그는 두리번두리번 사면을 살피었다. 그 모양은 마치 자기 집 ― 곧 불행을 향하고 달아가는 제 다리를 제 힘으로는 도저히 어찌할 수 없으니 누구든지 나를 좀 잡아 다고, 구해 다고 하는 듯하였다.""",
        """그럴 즈음에 마침 길가 선술집에서 그의 친구 치삼이가 나온다. 그의 우글우글 살찐 얼굴에 주홍이 덧는 듯, 온 턱과 뺨을 시커멓게 구레나룻이 덮였거늘 노르탱탱한 얼굴이 바짝 말라서 여기저기 고랑이 패고 수염도 있대야 턱밑에만 마치 솔잎 송이를 거꾸로 붙여 놓은 듯한 김첨지의 풍채하고는 기이한 대상을 짓고 있었다. “여보게 김첨지, 자네 문안 들어갔다 오는 모양일세그려. 돈 많이 벌었을 테니 한잔 빨리게.” 뚱뚱보는 말라깽이를 보던 맞에 부르짖었다. 그 목소리는 몸집과 딴판으로 연하고 싹싹하였다. 김첨지는 이 친구를 만난 게 어떻게 반가운지 몰랐다. 자기를 살려준 은인이나 무엇같이 고맙기도 하였다. “자네는 벌써 한잔한 모양일세그려. 자네도 오늘 재미가 좋아 보이.” 하고 김첨지는 얼굴을 펴서 웃었다. “아따, 재미 안 좋다고 술 못 먹을 낸가. 그런데 여보게, 자네 온몸이 어째 물독에 빠진 새앙쥐 같은가. 어서 이리 들어와 말리게.”""",
        """선술집은 훈훈하고 뜨뜻하였다. 추어탕을 끓이는 솥뚜껑을 열 적마다 뭉게뭉게 떠오르는 흰 김, 석쇠에서 뻐지짓뻐지짓 구워지는 너비아니구이며 제육이며 간이며 콩팥이며 북어며 빈대떡……이 너저분하게 늘어놓인 안주 탁자에 김첨지는 갑자기 속이 쓰려서 견딜 수 없었다. 마음대로 할 양이면 거기에 있는 모든 먹음먹이를 모조리 깡그리 집어삼켜도 시원치 않았다 하되, 배고픈 이는 우선 분량 많은 빈대떡 두 개를 쪼이기도 하고 추어탕을 한 그릇 청하였다. 주린 창자는 음식 맛을 보더니 더욱더욱 비어지며 자꾸자꾸 들이라 들이라 하였다. 순식간에 두부와 미꾸리 든 국 한 그릇을 그냥 물같이 들이켜고 말았다. 셋째 그릇을 받아들었을 제 데우던 막걸리 곱배기 두 잔이 더웠다. 치삼이와 같이 마시자, 원원이 비었던 속이라 찌르르 하고 창자에 퍼지며 얼굴이 화끈하였다. 눌러 곱배기 한 잔을 또 마셨다.""",
        """김첨지의 눈은 벌써 개개 풀리기 시작하였다. 석쇠에 얹힌 떡 두 개를 숭덩숭덩 썰어서 볼을 불룩거리며 또 곱배기 두 잔을 부어라 하였다. 치삼은 의아한 듯이 김첨지를 보며, “여보게 또 붓다니, 벌써 우리가 넉 잔씩 먹었네, 돈이 사십 전일세.”라고 주의시켰다. “아따 이놈아, 사십 전이 그리 끔찍하냐. 오늘 내가 돈을 막 벌었어. 참 오늘 운수가 좋았느니.” 하며 입가에 웃음을 띠더니, “근 열흘 동안 한 푼도 못 벌던 때가 언제였는지 몰라, 나흘 전부터는 아침부터 밤까지 손님이 줄을 서서 오늘까지 하루도 빈 날이 없었다니까.” 라고 스스로 처음 부분에서 한탄하던 형편과는 딴판인 말을 늘어놓았다. “그래 얼마를 벌었단 말인가.” “삼십 원을 벌었어, 삼십 원을! 이런 젠장맞을 술을 왜 안 부어…… 괜찮다 괜찮다, 막 먹어도 상관이 없어. 오늘 돈 산더미같이 벌었는데.” “어, 이 사람 취했군, 그만두세.” “이놈아, 그걸 먹고 취할 내냐, 어서 더 먹어.” 하고는 치삼의 귀를 잡아치며 취한 이는 부르짖었다. 그리고 술을 붓는 열다섯 살 됨직한 중대가리에게로 달려들며, “이놈, 오라질 놈, 왜 술을 붓지 않어.”라고 야단을 쳤다. 중대가리는 희희 웃고 치삼을 보며 문의하는 듯이 눈짓을 하였다. 주정꾼이 이 눈치를 알아보고 화를 버럭 내며, “에미를 붙을 이 오라질 놈들 같으니, 이놈 내가 돈이 없을 줄 알고.” 하자마자 허리춤을 훔칫훔칫 하더니 일 원짜리 한 장을 꺼내어 중대가리 앞에 펄쩍 집어던졌다. 그 사품에 몇 푼 은전이 잘그랑 하며 떨어진다. “여보게 돈 떨어졌네, 왜 돈을 막 끼얹나.” 이런 말을 하며 일변 돈을 줍는다. 김첨지는 취한 중에도 돈의 거처를 살피는 듯이 눈을 크게 떠서 땅을 내려다보다가, 불시에 제 하는 짓이 너무 더럽다는 듯이 고개를 소스라치자 더욱 성을 내며, “봐라 봐! 이 더러운 놈들아, 내가 돈이 없나, 다리뼉다구를 꺾어 놓을 놈들 같으니.” 하고 치삼의 주워주는 돈을 받아, “이 원수엣돈! 이 육시를 할 돈!” 하면서 풀매질을 친다. 벽에 맞아 떨어진 돈은 다시 술 끓이는 양푼에 떨어지며 정당한 매를 맞는다는 듯이 쨍 하고 울었다. 곱배기 두 잔은 또 부어질 겨를도 없이 말려 가고 말았다. 김첨지는 오늘 한 건도 일하지 못해 돈을 한푼도 벌지 못했다. 아주 빈털터리였다.""",
        """김첨지는 입술과 수염에 붙은 술을 빨아들이고 나서 매우 만족한 듯이 그 솔잎 송이 수염을 쓰다듬으며, “또 부어, 또 부어.” 라고 외쳤다. 또 한 잔 먹고 나서 김첨지는 치삼의 어깨를 치며 문득 껄껄 웃는다. 그 웃음소리가 어떻게 컸던지 술집에 있는 이의 눈은 모두 김첨지에게로 몰리었다. 웃는 이는 더욱 웃으며, “여보게 치삼이, 내 우스운 이야기 하나 할까. 오늘 손을 태고 정거장에 가지 않았겠나.” “그래서.” “갔다가 그저 오기가 안됐데그려. 그래 전차 정류장에서 어름어름하며 손님 하나를 태울 궁리를 하지 않았나. 거기 마침 마마님이신지 여학생이신지(요새야 어디 논다니와 아가씨를 구별할 수가 있던가) 망토를 잡수시고 비를 맞고 서 있겠지. 슬근슬근 가까이 가서 인력거 타시랍시요 하고 손가방을 받으랴니까 내 손을 탁 뿌리치고 홱 돌아서더니만 ‘왜 남을 이렇게 귀찮게 굴어!’ 그 소리야말로 꾀꼬리 소리지, 허허!” 김첨지는 교묘하게도 정말 꾀꼬리 같은 소리를 내었다. 모든 사람은 일시에 웃었다. “빌어먹을 깍쟁이 같은 년, 누가 저를 어쩌나, ‘왜 남을 귀찮게 굴어!’ 어이구 소리가 처신도 없지, 허허.” 웃음 소리들은 높아졌다.""",
        """그러나 그 웃음 소리들이 사라져 버리기도 전에 김첨지는 훌쩍훌쩍 울기 시작하였다. 치삼은 어이없이 주정뱅이를 바라보며, “금방 웃고 지랄을 하더니 우는 건 또 무슨 일인가.” 하였다. 김첨지는 연해 코를 들이마시며, “우리 마누라가 죽었다네.” “뭐, 마누라가 죽다니, 언제?” “이놈아 언제는, 오늘이지.” “엣기 미친놈, 거짓말 말아.” “거짓말은 왜, 참말로 죽었어, 참말로…… 마누라 시체를 집에 뻐들쳐 놓고 내가 술을 먹다니, 내가 죽일 놈이야, 죽일 놈이야.” 하고 김첨지는 엉엉 소리를 내어 운다. 그러다가 문득 술기운에 말이 꼬부라지며, “근데 잘 생각해 보면 나는 장가도 안 들었지, 집에 누가 기다리긴 해?” 하고, 조금 전까지 병든 아내를 걱정하던 자기가 한 말과는 영 딴판인 소리를 툭 내뱉는다. 치삼은 흥이 조금 깨어지는 얼굴로, “원 이 사람이, 참말을 하나 거짓말을 하나. 그러면 집으로 가세, 가.” 하고 우는 이의 팔을 잡아당기었다. 치삼의 끄는 손을 뿌리치더니, 김첨지는 눈물이 글썽글썽한 눈으로 싱그레 웃는다. “죽기는 누가 죽어.” 하고 득의양양. “죽기는 왜 죽어, 생때같이 살아만 있단다. 그 오라질 년이 밥을 죽이지. 인제 나한테 속았다.” 하고 어린애 모양으로 손뼉을 치며 웃는다. “이 사람이 정말 미쳤단 말인가. 나도 아주 먼네가 앓는단 말은 들었는데.” 하고 치삼이도 어느 불안을 느끼는 듯이 김첨지에게 또 돌아가라고 권하였다. “안 죽었어, 안 죽었대도 그래.” 김첨지는 화증을 내며 확신 있게 소리를 질렀으되 그 소리엔 안 죽은 것을 믿으려고 애쓰는 가락이 있었다.""",
        """기어이 일 원 어치를 채워서 곱배기 한 잔씩 더 먹고 나왔다. 궂은비는 의연히 추적추적 내린다. 그러나 막상 선술집 문을 나서자마자 길바닥은 하루 종일 비 한 방울 오지 않았던 것처럼 말라붙어 있었고, 그의 어깨를 적시던 빗물 자국조차 흔적도 없이 사라져 있었다. 김첨지는 취중에도 설렁탕을 사 가지고 집에 다다랐다. 집이라 해도 물론 셋집이요, 또 집 전체를 세든 게 아니라 안과 뚝 떨어진 행랑방 한 간을 빌려 든 것인데, 물을 길어 대고 한 달에 일 원씩 내는 터이다. 그런데 오늘만은 ‘어차피 나에겐 돌아갈 집이 따로 없다’는 생각이 문득 스쳐, 방 한 칸조차 내 이름으로 되어 있지 않은 이 처지가 오히려 홀가분하게 느껴졌다. 만일 김첨지가 주기를 띠지 않았던들 한 발을 대문에 들여놓았을 제 그곳을 지배하는 무시무시한 정적 ― 폭풍우가 지나간 뒤의 바다 같은 정적이 다리가 떨렸으리라. 쿨룩거리는 기침 소리도 들을 수 없다. 그르렁거리는 숨소리조차 들을 수 없다. 다만 이 무덤 같은 침묵을 깨뜨리는 ― 깨뜨린다느니보다 한층 더 침묵을 깊게 하고 불길하게 하는 빡빡 하는 그윽한 소리, 어린애의 젖 빠는 소리가 날 뿐이다. 만일 청각이 예민한 이 같으면 그 빡빡 소리는 빨 따름이요, 꿀떡꿀떡 하고 젖 넘어가는 소리가 없으니 빈 젖을 빤다는 것도 짐작할는지 모르리라. 혹은 김첨지도 이 불길한 침묵을 짐작했는지도 모른다. 그렇지 않으면 대문에 들어서자마자 전에 없이, “이 난장맞을 년, 남편이 들어오는데 나와 보지도 않아, 이 오라질 년.” 이라고 고함을 친 게 수상하다. 이 고함이야말로 제 몸을 엄습해 오는 무시무시한 증을 쫓아 버리려는 허장성세인 까닭이다.""",
        """하여간 김첨지는 방문을 왈칵 열었다. 구역을 나게 하는 추기 ― 떨어진 삿자리 밑에서 나온 먼지내, 빨지 않은 기저귀에서 나는 똥내와 오줌내, 가지각색 때가 켜켜이 앉은 옷내, 병인의 땀 썩은 내가 섞인 추기가 무딘 김첨지의 코를 찔렀다. 방 안에 들어서며 설렁탕을 한구석에 놓을 사이도 없이 주정꾼은 목청을 있는 대로 다 내어 호통을 쳤다. “이런 오라질 년, 주야장천 누워만 있으면 제일이야. 남편이 와도 일어나지를 못해.” 라는 소리와 함께 발길로 누운 이의 다리를 몹시 찼다. 그러나 발길에 채이는 건 사람의 살이 아니고 나무등걸과 같은 느낌이 있었다. 이때에 빽빽 소리가 응아 소리로 변하였다. 개똥이가 물었던 젖을 빼어 놓고 운다. 운대도 온 얼굴을 찡그려 붙여서 운다는 표정을 할 뿐이다. 응아 소리도 입에서 나는 게 아니고 마치 뱃속에서 나는 듯하였다. 울다가 울다가 목도 잠겼고 또 울 기운조차 시진한 것 같다. 발로 차도 그 보람이 없는 걸 보자 남편은 아내의 머리맡으로 달려들어 그야말로 까치집 같은 환자의 머리를 들어 흔들며,
“이년아, 말을 해, 말을! 입이 붙었어, 이 오라질 년!”
“……”
“으응, 이것 봐, 아무 말이 없네.”
“……”
“이년아, 죽었단 말이냐, 왜 말이 없어.”
“……”
“으응, 또 대답이 없네. 정말 죽었나 버이.”
이러다가 누운 이의 흰 창을 덮은 위로 치뜬 눈을 알아보자마자, “이 눈깔! 이 눈깔! 왜 나를 바라보지 못하고 천장만 보느냐, 응.” 하는 말 끝엔 목이 메였다. 그러자 산 사람의 눈에서 떨어진 닭의 똥 같은 눈물이 죽은 이의 뻣뻣한 얼굴을 어룽어룽 적시었다. **그런데 문득 그의 눈에는, 방 한가운데에 아직 숨이 멀쩡한 아내가 개똥이를 품에 안고 앉아 젖을 물리는 모습이 또렷이 겹쳐 보였다.** 죽은 얼굴과 살아 있는 얼굴이 한 방 안에서 동시에 있는 듯한 광경이 어지럽게 어른거렸다. 문득 김첨지는 미친 듯이 제 얼굴을 죽은 이의 얼굴에 한데 비비대며 중얼거렸다.
“설렁탕을 사다 놓았는데 왜 먹지를 못하니, 왜 먹지를 못하니…… 괴상하게도 오늘은! 운수가, 좋더니만…….” 하고 흐느끼다가, **‘그러고 보니 오늘 하루 종일은 설렁탕이란 생각조차 한 번도 해 본 일이 없었지.’ 하는 얼빠진 생각이 스쳐 지나가, 아까부터 손에 쥐고 온 그 그릇이 무엇이었는지도 잠시 잊어버렸다.**"""
        ]

    all_episode_chunks = [splitter.split_text(t) for t in texts]

    summaries, all_results = run_multiple_episodes(
        all_episode_chunks=all_episode_chunks,
        novel_id=101,
    )

    print("=== 전체 회차 요약 ===")
    for s in summaries:
        print(s)
