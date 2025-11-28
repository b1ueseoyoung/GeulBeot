import os
import json
import time
from typing import List, Dict, Any, Optional, Set

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS

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
PER_CHUNK_DELAY = 0.0

# 벡터 검색 Top-K
SEARCH_TOP_K = 5           # 설정 검색
SEARCH_TOP_K_CONTEXT = 3   # 맥락 검색

# 시스템 프롬프트 캐시
LORE_KEEPER_SYSTEM_PROMPT = ""

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

# ==========================
# OpenAI / LLM 클라이언트 (재사용)
# ==========================

# OpenAI API Key 설정


embeddings = OpenAIEmbeddings()

LLM_CLASSIFIER = ChatOpenAI(model="gpt-4o-mini", temperature=0)     # 청크 타입 분류용
LLM_FACT_EXTRACTOR = ChatOpenAI(model="gpt-4o", temperature=0)      # 사실 추출용
LLM_AGENT_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)    # ReAct 에이전트용

# ==========================
# DB 스키마 Enum 유틸
# ==========================

ITEM_TYPES = {"FACT", "RULE", "EXCEPTION"}
CATEGORIES = {
    "PHY_STATUS", "PHY_TRAIT", "ABILITY", "ITEM",
    "RELATION", "LOCATION", "WORLD_SETTING",
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
) -> Dict[str, Any]:
    """lore_items 스키마에 맞춰 raw fact를 정규화"""
    return {
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


def add_to_full_story_db_vectorstore(chunk_data_list: List[str]):
    """원본 chunk_data를 Full_Story_DB 벡터 스토어에 추가"""
    if not chunk_data_list:
        return

    global _full_story_db_vectordb

    texts = chunk_data_list
    metadatas = [
        {"chunk_index": i, "source": "processed_episode"}
        for i in range(len(chunk_data_list))
    ]

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


def add_to_current_chunk_vectorstore(chunks: List[str], episode_seq: int = DEFAULT_EPISODE_SEQ):
    """
    현재 회차 원문 청크를 한 번에 FAISS 벡터 스토어에 적재.
    - 회차 시작 시 1회 호출 권장.
    """
    if not chunks:
        return

    global _current_chunk_vectordb

    metadatas = [
        {"chunk_index": i, "source_seq": str(episode_seq)}
        for i in range(len(chunks))
    ]

    if _current_chunk_vectordb is None:
        print(f"[Current_Chunks] 새 인덱스 생성, 총 {len(chunks)}개 적재")
        _current_chunk_vectordb = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=metadatas,
        )
    else:
        print(f"[Current_Chunks] 기존 인덱스에 {len(chunks)}개 추가")
        _current_chunk_vectordb.add_texts(texts=chunks, metadatas=metadatas)


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


def save_full_story_to_file(chunks: List[str], path: str = FULL_STORY_DB_FILE, episode_seq: int = DEFAULT_EPISODE_SEQ):
    if not chunks:
        return
    df_new = pd.DataFrame(
        [{"text": ck, "source_seq": episode_seq, "chunk_index": i} for i, ck in enumerate(chunks)]
    )
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
                    add_to_full_story_db_vectorstore(texts)
                    print(f"[Load] full_story_db {len(texts)}개 로드 및 벡터 반영 (새로 생성)")
                else:
                    print(f"[Load] full_story_db {len(texts)}개 로드 (기존 벡터스토어 사용)")
        except Exception as e:
            print(f"[Load] full_story_db 로드 실패: {e}")

    _state_loaded = True


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
    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K)

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

    return "[Current_DB 검색 결과 (임시 설정)]\n" + "\n\n".join(lines)


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
    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K)

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

    return "[Lore_DB 검색 결과 (정규화된 설정)]\n" + "\n\n".join(lines)


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
    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K_CONTEXT)

    if not docs:
        return "[Full_Story_DB에서 관련 맥락 없음]"

    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        lines.append(
            f"[Full_Story_DB] chunk_index={meta.get('chunk_index', '?')}, source={meta.get('source', '?')}\n"
            f"{i}. {doc.page_content}"
        )
    return "[Full_Story_DB 검색 결과 (원본 맥락)]\n" + "\n\n".join(lines)


@tool
def search_current_chunks(query: str) -> str:
    """
    [Current_Chunks 검색 도구]
    현재 회차의 원문 청크를 임시 벡터 스토어에서 검색합니다.
    """
    vectordb = get_current_chunk_vectordb()
    if vectordb is None:
        return "[Current_Chunks가 비어있습니다]"

    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K_CONTEXT)

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
    return "[Current_Chunks 검색 결과 (원문)]\n" + "\n\n".join(lines)


@tool
def get_current_db_settings() -> str:
    """
    [Current_DB 조회 도구]
    현재 회차에서 확정된 설정들을 조회합니다.
    """
    if not current_story_db:
        return "[현재 회차에서 확정된 설정이 아직 없습니다]"

    lines = []
    for item in current_story_db[-20:]:  # 최근 20개만
        lines.append(
            f"- [{item['item_type']}/{item['category']}] "
            f"{item['subject']} -> {item['effect']}\n"
            f"  [source_seq={item.get('source_seq', '?')}] 원문: {item['text']}"
        )

    return "[Current_DB 설정들]\n" + "\n".join(lines)


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
        "A: 사실/설정 (세계관, 인물/아이템 상태, 사건, 배경)\n"
        "B: 감정/내면 (심리, 내면 독백)\n"
        "C: 대화 (직접 화법)\n"
        "D: 단순 서술 (묘사, 의성어 등)\n"
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
def extract_facts_from_chunk(chunk: str) -> str:
    """
    [사실 추출 도구]
    lore_items 스키마를 따르는 JSON 배열을 추출합니다.
    TTYPE_A/B/C로 분류된 청크에 대해서만 호출.
    """
    llm = LLM_FACT_EXTRACTOR

    system_msg = (
        "너는 웹소설 설정 추출 AI야.\n"
        "주어진 문장에서 lore_items 스키마를 추출해:\n"
        "item_type: FACT/RULE/EXCEPTION\n"
        "category: PHY_STATUS/PHY_TRAIT/ABILITY/ITEM/RELATION/LOCATION/WORLD_SETTING\n"
        "target_group: GLOBAL/RACE/CLASS/INDIVIDUAL\n"
        "chunk_type: TYPE_A/TYPE_B/TYPE_C\n"
        "subject, condition, effect, text를 포함한 JSON 배열만 출력.\n"
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
def save_to_current_db(chunk: str, facts_json: str) -> str:
    """
    [Current_DB 저장 도구]
    충돌이 없는 설정을 Current_DB에 저장합니다.
    """
    try:
        facts = json.loads(facts_json) if facts_json else []
        normalized = [
            _build_lore_item(fact, chunk, source_seq=CURRENT_EPISODE_SEQ)
            for fact in facts
        ]
        current_story_db.extend(normalized)
        add_to_current_db_vectorstore(normalized)
        return f"✓ Current_DB에 {len(normalized)}개 저장됨"
    except Exception as e:
        return f"✗ 저장 실패: {str(e)}"


@tool
def report_conflict_to_db(
    chunk: str,
    conflict_type: str,
    reason: str,
    facts_json: str = "[]",
    conflicting_text: str = "",
) -> str:
    """
    [충돌 리포트 도구]
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
            "episode_seq": CURRENT_EPISODE_SEQ,
            "chunk_index": CURRENT_CHUNK_INDEX,
            "input_text": chunk,
            "is_conflict": True,
            "conflict_type": normalized_type,
            "reason": reason,
            "evidence": conflicting_text or reason,
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


# ==========================
# Lore Keeper 에이전트 생성
# ==========================

def create_lore_keeper_agent(model_name: str = "gpt-4o-mini"):
    """Lore Keeper 에이전트 생성"""
    tools = [
        search_lore_db,
        search_current_db,
        search_current_chunks,
        get_current_db_settings,
        classify_chunk_type,
        extract_facts_from_chunk,
        save_to_current_db,
        report_conflict_to_db,
    ]

    # 기본은 재사용 LLM
    llm = LLM_AGENT_MODEL if model_name == "gpt-4o-mini" else ChatOpenAI(model=model_name, temperature=0)

    workflow = (
        "1. classify_chunk_type으로 A/B/C/D 분류\n"
        "2. D 타입이면 PASS (작업 종료)\n"
        "3. A/B/C 타입이면:\n"
        "   a. extract_facts_from_chunk로 사실 추출\n"
        "   c. search_lore_db (이전 회차 정규화된 설정 검색)\n"
        "   d. search_current_db (현재 회차 임시 설정 검색)\n"
        "   e. (필요할 경우) search_current_chunks (현재 회차 원문 맥락 검색)\n"
        "   f. get_current_db_settings (현재 회차 설정 조회)\n"
        "   g. 추출한 사실과 기존 설정 비교 (충돌 검사)\n"
        "   h. 충돌 있으면 report_conflict_to_db, 없으면 save_to_current_db\n"
    )

    system_prompt = f"""너는 웹소설 설정 관리 AI 에이전트 'Lore Keeper'야.

**작업 절차:**
{workflow}

**충돌 판정 기준:**
- Hard Conflict: 죽은 인물 재등장, 세계관 핵심 규칙 위반, 시간 모순
- Soft Conflict: 캐릭터 성격/말투 변화, 감정 흐름 이상

**충돌 리포트 작성 규칙:**
- report_conflict_to_db를 호출할 때:
  - conflict_type은 "Hard Conflict" 또는 "Soft Conflict"를 포함한 문자열로 넘겨.
  - reason에는 반드시 다음 정보를 포함해:
    * 현재 청크의 어떤 설정(문장/subject/effect)이
    * 어느 DB(Lore_DB / Current_DB / Full_Story_DB)의 어떤 설정(subject/effect/source_seq 등)과
    * 왜 충돌하는지 (시간, 상태, 규칙 위반 등)을 한국어로 명확히 서술.
  - conflicting_text에는 실제로 충돌하는 '기존 설정이나 원문 일부'를 그대로 붙여넣어.
- search_* 도구의 출력에 있는 source_seq, chunk_index 정보를 적극적으로 사용해.

**최종 응답 형식(반드시 이 형식을 그대로 따를 것):**
- 네가 모든 tool 호출과 판단을 마친 뒤, 사용자에게 보내는 마지막 답변은
  아래 2줄 형식을 정확히 지켜서 출력해야 한다. 다른 형태로 말하지 마.

1) "충돌 여부: 충돌"  또는  "충돌 여부: 충돌 없음"
2) "이후 동작: <한 줄 요약>"
   - 예시:
     - "이후 동작: Current_DB에 설정 저장 (추후 Lore_DB 반영 후보)"
     - "이후 동작: Conflict_DB에만 기록, Lore_DB / Full_Story_DB 반영 보류"
     - "이후 동작: TYPE_D 서술로 판단하여 저장/충돌 없이 PASS"

- 위 3줄 외에 불필요한 문장, 마크다운, 리스트, 설명은 출력하지 마.

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
    """에이전트로 청크 하나 처리"""
    global CURRENT_CHUNK_INDEX
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
        prev_current = len(current_story_db)
        prev_conflicts = len(conflict_db)

        result = agent_graph.invoke(
            {"messages": messages_payload},
            {"recursion_limit": 20}
        )

        messages = result["messages"]
        output = messages[-1].content

        print(f"\n[최종 판정] {output[:150]}...")

        newly_saved = current_story_db[prev_current:]
        newly_conflicted = conflict_db[prev_conflicts:]

        chunk_type = newly_saved[0].get("chunk_type", "TYPE_A") if newly_saved else "UNKNOWN"
        is_conflict = bool(newly_conflicted)
        conflict_type = newly_conflicted[0]["conflict_type"] if newly_conflicted else "None"
        conflicting_text = newly_conflicted[0].get("conflicting_text", "") if newly_conflicted else ""

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
    global CURRENT_EPISODE_SEQ
    CURRENT_EPISODE_SEQ = episode_seq

    load_persistent_state()
    reset_current_episode_state()

    print("\n" + "="*60)
    print(f"수동 회차 처리 시작 (episode_seq={episode_seq}, chunks={len(chunks)})")
    print("="*60 + "\n")

    # 벡터 스토어 준비 (lore/full 검색용)
    get_lore_db_vectordb()
    get_full_story_db_vectordb()

    # 현재 회차 원문 청크 → FAISS에 한 번에 적재
    add_to_current_chunk_vectorstore(chunks, episode_seq=episode_seq)

    agent = create_lore_keeper_agent()
    results = []

    for i, chunk in enumerate(chunks):
        res = process_chunk_with_agent(agent, chunk, index=i)
        results.append(res)
        if PER_CHUNK_DELAY > 0:
            time.sleep(PER_CHUNK_DELAY)

    # 최종 요약 및 후속 동작
    if conflict_db:
        print(f"\n✗ 충돌 {len(conflict_db)}건 감지 - Lore_DB / Full_Story_DB 반영 보류")
        summary = {
            "status": "conflict",
            "episode_seq": episode_seq,
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
        add_to_full_story_db_vectorstore(chunks)
        save_full_story_to_file(chunks, full_story_path, episode_seq=episode_seq)

        summary = {
            "status": "ok",
            "episode_seq": episode_seq,
            "chunk_count": len(chunks),
            "conflict_count": 0,
            "saved_lore_items": len(current_story_db),
            "action": "Lore_DB / Full_Story_DB 및 벡터 스토어에 저장",
        }

    # 콘솔용 결과 요약
    print("\n=== Lore Keeper 결과 요약 ===")
    print(f"- 회차: {summary.get('episode_seq')}")
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
            epi = c.get("episode_seq", "?")
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


# ==========================
# 예시 main (직접 실행 시)
# ==========================

if __name__ == "__main__":
    text = """쌍두취 행진곡 가을 학기가 되자, ○○일보사에서 주최하는 학생계몽운동에 참가하였던 대원들이 돌아왔다. 오늘 저녁은 각처에서 모여든 대원들을 위로하는 다과회가 그 신문사 누상에서 열린것이다.
 오륙백 명이나 수용할 수 있는 대강당에는 전 조선의 방방곡곡으로 흩어져서 한여름 동안 땀을 흘려 가며 활동한 남녀 대원들로 빈틈없이 들어찼다.
 폭양에 그을은 그들의 시커먼 얼굴! 큰 박덩이만큼씩 한 전등이 드문드문하게 달린 천장에서 내리비치는 불빛이 휘황할수록, 흰 벽을 등지고 앉은 그네들의 얼굴은 더한층 검어 보인다.
 만호 장안의 별처럼 깔린 등불이 한눈에 내려다보이도록 사방의 유리창을 활짝 열 어제 쳤건만, 건장한 청년들의 코와 몸에서 풍기는 훈김이 우거진 콩밭 속에를 들어간 것만치 나후 끈 후끈 끼친다.
 정각이 되자 P학당의 취주악대는 코넷, 트럼본 같은 번쩍거리는 악기를 들고 연단 앞줄에가 벌려 선다. 지휘자가 손을 내젓는 대로 힘차게 연주하는 것은 유명한 독일 사람의 작곡인 쌍두취 행진곡(雙頭鷲行進曲)이다. 그 활발하고 장쾌한 멜로디는 여러 사람의 심장까지 울리면서 장내의 공기를 진동시킨다.
 악대의 연주가 끝난 다음에, 사회자인 이 신문사의 편집국장이 안경을 번득이며 점잖은 걸음걸이로 단 위에 나타났다.
 "에― 아직 개학을 아니 헌 학교도 있어서 미처 올라오지 못한 대원이 많을 줄 알었습니다.
 그런데 뜻밖에 이처럼 성황을 이루어서 장소가 매우 협착한 까닭에, 여러분끼리 서로간 친하는 기회를 드리려는 다과회가 무슨 강연회처럼 되었습니다."
 하고 일장의 인사를 베푼 뒤에 으흠으흠 하고 헛기침을 해서 목소리를 가다듬더니,
 "금년에는 여러 가지로 지장이 많았는데도 불구하고 작년보다도 거진 곱절이나 되는 놀라울 만한 성적을 보게 됐습니다. 이것은 오직 동족을 사랑하는 여러분의 열성과, 문맹을 한 사람이라도 더 물리치려는 헌신적 노력의 결과인 것이 물론입니다. 그러므로 주 최자측으로서 여러분의 수고를 감사할 뿐 아니라, 우리 계몽운동의 장래를 위해서 경축하기를 마지않는 바입니다."
 처음에는 늦게 들어오는 사람들 때문에 수성수성하던 장내가 인제는 기침 소리 하나 없이 조용해졌다."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=30,
        separators=[
            r"(?<=[.?!])\s+",
            "\n",
            "\n\n",
        ],
        is_separator_regex=True,
        length_function=len,
    )

    chunks = splitter.split_text(text)

    print(f"len(chunks): {len(chunks)}")
    summary, results = run_manual_episode(
        chunks,
        episode_seq=1,
        clear_after=False,
        conflict_log_path="conflicts.jsonl"
    )
    print(summary)
