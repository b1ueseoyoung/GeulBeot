import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.vectorstores import FAISS

# ==========================
# 전역 상수 / 설정
# ==========================

# 파일/디렉토리 경로
LORE_DB_VECTOR_DIR = "./faiss_lore_db"              # Lore_DB (정규화 설정) 벡터 스토어
FULL_STORY_DB_VECTOR_DIR = "./faiss_full_story_db"  # Full_Story_DB (원본 청크) 벡터 스토어
LORE_DB_FILE = "./lore_db.jsonl"
FULL_STORY_DB_FILE = "./full_story_db.jsonl"
CONFLICT_DB_FILE = "./conflict_db.jsonl"

# 회차 기본값
DEFAULT_EPISODE_SEQ = 1

# 청크 처리 사이 딜레이 (rate limit 대비용, 기본 0: 없음)
PER_CHUNK_DELAY = 0.3

# 벡터 검색 Top-K (개선: 5 → 10으로 증가)
SEARCH_TOP_K = 10          # 설정 검색
SEARCH_TOP_K_CONTEXT = 5   # 맥락 검색

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

_lore_db_vectordb: Optional[FAISS] = None       # Lore_DB 벡터 스토어
_full_story_db_vectordb: Optional[FAISS] = None # Full_Story_DB 벡터 스토어
_current_story_vectordb: Optional[FAISS] = None # Current_DB 임시 벡터 스토어
_current_chunk_vectordb: Optional[FAISS] = None # Current_Chunks 임시 벡터 스토어 (FAISS)

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

# 보안: 키는 코드에 절대 하드코딩하지 말고 환경변수로만 주입하세요.
# macOS/zsh 예: export OPENAI_API_KEY="sk-..."
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please export it in your shell environment "
        "(e.g., export OPENAI_API_KEY='sk-...') and rerun."
    )

embeddings = OpenAIEmbeddings()

LLM_CLASSIFIER = ChatOpenAI(model="gpt-4o-mini", temperature=0)
LLM_FACT_EXTRACTOR = ChatOpenAI(model="gpt-4o", temperature=0)
LLM_AGENT_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# o3-mini는 reasoning 모델이라 temperature 파라미터를 지원하지 않음
CONFLICT_JUDGE_LLM = ChatOpenAI(model="o3-mini", max_retries=5)

# ==========================
# DB 스키마 Enum 유틸
# ==========================

ITEM_TYPES = {"FACT", "RULE", "EXCEPTION"}

# ✅ 수정: 프롬프트에서 사용하는 EVENT/TIME/WEATHER/GUESS를 enum에 추가 (오염 방지 핵심)
CATEGORIES = {
    "PHY_STATUS", "PHY_TRAIT", "ABILITY", "ITEM",
    "RELATION", "LOCATION", "WORLD_SETTING", "EMOTION",
    "EVENT", "TIME", "WEATHER", "GUESS"
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
    nid = novel_id if novel_id is not None else raw_fact.get("novel_id", CURRENT_NOVEL_ID)
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
        print(f"[Lore_DB_VectorStore] Load existing vector store: {LORE_DB_VECTOR_DIR}")
        _lore_db_vectordb = FAISS.load_local(
            LORE_DB_VECTOR_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return _lore_db_vectordb

    return None


def get_full_story_db_vectordb() -> Optional[FAISS]:
    """Full_Story_DB 전용 벡터 스토어 (원본 chunk_data)"""
    global _full_story_db_vectordb

    if _full_story_db_vectordb is not None:
        return _full_story_db_vectordb

    if os.path.exists(FULL_STORY_DB_VECTOR_DIR):
        print(f"[Full_Story_DB_VectorStore] Load existing vector store: {FULL_STORY_DB_VECTOR_DIR}")
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

    if _lore_db_vectordb is None:
        print("[Lore_DB_VectorStore] Create a new FAISS index and load initial data")
        _lore_db_vectordb = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
    else:
        print("[Lore_DB_VectorStore] Append data to the existing index")
        _lore_db_vectordb.add_texts(texts=texts, metadatas=metadatas)

    _lore_db_vectordb.save_local(LORE_DB_VECTOR_DIR)
    print(f"[Lore_DB_VectorStore] Saved {len(texts)} lore_items as vectors.")


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
            "source_seq": episode_seq,
            "chunk_type": ct,
        }
        metadatas.append(meta)

    if _full_story_db_vectordb is None:
        print("[Full_Story_DB_VectorStore] Create a new FAISS index and load initial data")
        _full_story_db_vectordb = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
    else:
        print("[Full_Story_DB_VectorStore] Append data to the existing index")
        _full_story_db_vectordb.add_texts(texts=texts, metadatas=metadatas)

    _full_story_db_vectordb.save_local(FULL_STORY_DB_VECTOR_DIR)
    print(f"[Full_Story_DB_VectorStore] Saved {len(chunk_data_list)} chunk_data as vectors.")


# ==========================
# Current_Chunks (FAISS, 회차 단위 임시 벡터 스토어)
# ==========================

def get_current_chunk_vectordb() -> Optional[FAISS]:
    """현재 회차 원문 청크용 임시 벡터 스토어(FAISS)를 반환합니다."""
    return _current_chunk_vectordb


def add_to_current_chunk_vectorstore(chunk: str, episode_seq: int = CURRENT_EPISODE_SEQ):
    """지금 청크 하나를 Current_Chunks 벡터스토어에 적재."""
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
        print("[Current_Chunks] Create a new index, load 1 item total")
        _current_chunk_vectordb = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
    else:
        print(f"[Current_Chunks] Append 1 item to the existing index (chunk_index={CURRENT_CHUNK_INDEX})")
        _current_chunk_vectordb.add_texts(texts=texts, metadatas=metadatas)


# ==========================
# Current_DB (현재 회차 정규화 설정용 벡터 스토어)
# ==========================

def get_current_story_vectordb() -> Optional[FAISS]:
    """Current_DB 전용 벡터 스토어 (회차 단위 임시, non-persist)"""
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
        print(f"[Current_DB_VectorStore] Create a new index, load {len(texts)} items")
        _current_story_vectordb = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
    else:
        print(f"[Current_DB_VectorStore] Append {len(texts)} items to the existing index")
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
    print(f"[Conflict_Log] Logged {len(conflicts)} items → {path}")


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
    print(f"[Lore_DB_File] Saved/appended {len(items)} items → {path}")


def save_full_story_to_file(
    chunks: List[str],
    path: str = FULL_STORY_DB_FILE,
    episode_seq: int = DEFAULT_EPISODE_SEQ,
    chunk_types: Optional[List[str]] = None,
    novel_id: Optional[int] = None
):
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
            "chunk_type": ct,
        }
        rows.append(row)
    df_new = pd.DataFrame(rows)
    df_old = _load_dataframe(path)
    if df_old is not None:
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    _write_dataframe(df_new, path)
    print(f"[Full_Story_File] Saved/appended {len(chunks)} items → {path}")


def load_persistent_state():
    """
    json/jsonl에서 lore_db/full_story_db를 로드하고,
    벡터 디렉토리가 없을 때만 초기 임베딩을 수행.
    """
    global _state_loaded, _full_story_db_vectordb, _lore_db_vectordb
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
                print(f"[Load] Loaded lore_db ({len(lore_db)} items) and built vectors (new)")
            else:
                print(f"[Load] Loaded lore_db ({len(lore_db)} items) (using existing vector store)")
        except Exception as e:
            print(f"[Load] lore_db load failed: {e}")

    # Full_Story_DB
    if os.path.exists(FULL_STORY_DB_FILE):
        try:
            df = _load_dataframe(FULL_STORY_DB_FILE)
            if df is not None and "text" in df.columns:
                texts = df["text"].tolist()
                full_story_db.extend(texts)

                # (선택) 벡터스토어가 없다면 새로 구축
                if texts and not full_vectordb_exists:
                    records = df.to_dict("records")
                    meta_list = []
                    for i, r in enumerate(records):
                        meta_list.append(
                            {
                                "novel_id": r.get("novel_id", 0),
                                "source_seq": r.get("source_seq", DEFAULT_EPISODE_SEQ),
                                "chunk_index": r.get("chunk_index", i),
                                "chunk_type": r.get("chunk_type", "TYPE_D"),
                                "source": "load",
                            }
                        )
                    _full_story_db_vectordb = FAISS.from_texts(
                        texts=texts,
                        embedding=embeddings,
                        metadatas=meta_list,
                    )
                    _full_story_db_vectordb.save_local(FULL_STORY_DB_VECTOR_DIR)
                    print(f"[Load] Loaded full_story_db ({len(texts)} items) and built vectors (new)")
                else:
                    print(f"[Load] Loaded full_story_db ({len(texts)} items) (using existing vector store)")
        except Exception as e:
            print(f"[Load] full_story_db load failed: {e}")

    _state_loaded = True


# ==========================
# 에이전트 수행 결과 처리
# ==========================

def are_conflicting_values(val1: str, val2: str) -> bool:
    """두 값이 충돌하는지 확인 (주로 숫자 비교)"""
    # 숫자 추출 (달러, 나이 등)
    def extract_numbers(text: str):
        numbers = re.findall(r'\$?(\d+(?:\.\d+)?)', text)
        return [float(n) for n in numbers]
    
    nums1 = extract_numbers(val1)
    nums2 = extract_numbers(val2)
    
    # 둘 다 숫자가 있고, 다른 경우
    if nums1 and nums2 and nums1[0] != nums2[0]:
        return True
    
    # 텍스트 기반 충돌
    conflicting_pairs = [
        ("long", "short"), ("sold", "not sold"), ("sold", "never sold"),
        ("yes", "no"), ("true", "false"),
    ]
    
    val1_lower, val2_lower = val1.lower(), val2.lower()
    for word1, word2 in conflicting_pairs:
        if (word1 in val1_lower and word2 in val2_lower) or (word2 in val1_lower and word1 in val2_lower):
            return True
    
    return False


def check_internal_conflicts(facts: list, chunk: str) -> dict:
    """같은 청크에서 추출된 facts 간 모순 확인"""
    if not facts or len(facts) < 2:
        return None
    
    for i, fact1 in enumerate(facts):
        for fact2 in facts[i+1:]:
            subj1 = str(fact1.get('subject', '')).lower().strip()
            subj2 = str(fact2.get('subject', '')).lower().strip()
            
            # Subject가 비슷하거나 같은 경우
            if subj1 and subj2 and (subj1 == subj2 or subj1 in subj2 or subj2 in subj1):
                cat1, cat2 = fact1.get('category', ''), fact2.get('category', '')
                
                if cat1 == cat2:
                    eff1, eff2 = str(fact1.get('effect', '')), str(fact2.get('effect', ''))
                    
                    if are_conflicting_values(eff1, eff2):
                        return {
                            "is_conflict": True,
                            "conflict_type": "Hard Conflict",
                            "reason": f"Internal contradiction: '{eff1}' vs '{eff2}' for subject '{subj1}'",
                            "facts": [fact1, fact2],
                            "conflicting_text": fact2.get('text', chunk),
                        }
    return None


def save_to_current_db(chunk: str, facts_json: str) -> str:
    """[Current_DB 저장] 충돌이 없는 설정을 Current_DB에 저장합니다."""
    try:
        facts = json.loads(facts_json) if facts_json else []

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
        return f"✓ Saved {len(normalized)} items to Current_DB (duplicates/guesses excluded)"
    except Exception as e:
        return f"✗ Save failed: {str(e)}"


def report_conflict_to_db(
    chunk: str,
    conflict_type: str,
    reason: str,
    facts_json: str = "[]",
    conflicting_text: str = "",
) -> str:
    """[충돌 리포트] 감지된 설정 충돌을 Conflict_DB에 저장합니다."""
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

        msg = f"✗ Detected: {normalized_type}"
        if conflicting_text:
            msg += f"\n  Existing-setting snippet: {conflicting_text[:50]}..."
        return msg
    except Exception as e:
        return f"✗ Save failed: {str(e)}"


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
                try:
                    judge_result = json.loads(raw)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[Warning] judge_conflict returned invalid JSON: {raw[:100]}")
                    # Fallback: assume no conflict if JSON parsing fails
                    judge_result = {"is_conflict": False, "conflict_type": "None", "reason": "JSON parse error"}
                    pass

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
    """[Current_DB 검색 도구] 현재 회차에서 정규화된 설정 검색."""
    if not current_story_db:
        return "[Current_DB is empty]"

    vectordb = get_current_story_vectordb()
    if vectordb is None:
        return "[Current_DB vector store has not been created yet]"
    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K, filter={"novel_id": CURRENT_NOVEL_ID})

    if not docs:
        return "[No related settings found in Current_DB]"

    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        lines.append(
            f"{i}. [Current_DB] [{meta.get('item_type')}/{meta.get('category')}] "
            f"{meta.get('subject')} -> {meta.get('effect')}\n"
            f"   [source_seq={meta.get('source_seq', '?')}] Original: {meta.get('text', '')}"
        )

    result = "[Current_DB Search Results (temporary settings)]\n" + "\n\n".join(lines)
    SEARCH_CONTEXT_BY_CHUNK[CURRENT_CHUNK_INDEX]["current"] = result
    return result


@tool
def search_lore_db(query: str) -> str:
    """[Lore_DB 검색 도구] 이전 회차들에서 확정된 설정 검색."""
    if not lore_db:
        return "[Lore_DB is empty]"

    vectordb = get_lore_db_vectordb()
    if vectordb is None:
        return "[Lore_DB vector store has not been created yet]"
    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K, filter={"novel_id": CURRENT_NOVEL_ID})

    if not docs:
        return "[No related settings found in Lore_DB]"

    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        lines.append(
            f"{i}. [Lore_DB] [{meta.get('item_type')}/{meta.get('category')}] "
            f"{meta.get('subject')} -> {meta.get('effect')}\n"
            f"   [source_seq={meta.get('source_seq', '?')}] Original: {meta.get('text', '')}"
        )

    result = "[Lore_DB Search Results (normalized settings)]\n" + "\n\n".join(lines)
    SEARCH_CONTEXT_BY_CHUNK[CURRENT_CHUNK_INDEX]["lore"] = result
    return result


@tool
def search_full_story_db(query: str) -> str:
    """[Full_Story_DB 검색 도구] 이전 회차들의 원본 chunk_data 검색 (맥락 확인용)."""
    if not full_story_db:
        return "[Full_Story_DB is empty]"

    vectordb = get_full_story_db_vectordb()
    if vectordb is None:
        return "[Full_Story_DB vector store has not been created yet]"
    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K_CONTEXT, filter={"novel_id": CURRENT_NOVEL_ID})

    if not docs:
        return "[No related context found in Full_Story_DB]"

    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        lines.append(
            f"[Full_Story_DB] chunk_index={meta.get('chunk_index', '?')}, source={meta.get('source', '?')}\n"
            f"{i}. {doc.page_content}"
        )
    result = "[Full_Story_DB Search Results (raw context)]\n" + "\n\n".join(lines)
    SEARCH_CONTEXT_BY_CHUNK[CURRENT_CHUNK_INDEX]["full"] = result
    return result


@tool
def search_current_chunks(query: str) -> str:
    """[Current_Chunks 검색 도구] 현재 회차의 원문 청크를 검색."""
    vectordb = get_current_chunk_vectordb()
    if vectordb is None:
        return "[Current_Chunks is empty]"

    docs = vectordb.similarity_search(query, k=SEARCH_TOP_K_CONTEXT, filter={"novel_id": CURRENT_NOVEL_ID})

    if not docs:
        return "[No related context found in Current_Chunks]"

    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        lines.append(
            f"{i}. [Current_Chunks] (chunk_index={meta.get('chunk_index', '?')}, "
            f"source_seq={meta.get('source_seq', '?')})\n"
            f"   {doc.page_content}"
        )
    result = "[Current_Chunks Search Results (raw text)]\n" + "\n\n".join(lines)
    SEARCH_CONTEXT_BY_CHUNK[CURRENT_CHUNK_INDEX]["chunks"] = result
    return result


@tool
def classify_chunk_type(chunk: str) -> str:
    """[청크 타입 분류 도구] A(설정), B(감정), C(대화), D(단순 서술)로 분류."""
    llm = LLM_CLASSIFIER

    system_msg = (
        "You are an AI that classifies web-novel chunks.\n"
        "Please classify the given text into exactly one of A/B/C/D:\n"
        "A: Facts/Worldbuilding (world setting, character/item state, events, background, space, time, season, weather)\n"
        "B: Emotion/Inner state (psychology, inner monologue)\n"
        "C: Dialogue (direct speech)\n"
        "D: Plain narration (low-information description, sound effects, swearing, metaphors, etc.)\n"
        "Output ONLY one single character: A, B, C, or D."
    )

    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Please classify this:\n{chunk}"}
    ])

    text = resp.content.strip().upper()
    for ch in ["A", "B", "C", "D"]:
        if ch in text:
            return ch
    return "D"


@tool
def extract_facts_from_chunk(chunk: str, chunk_type: str) -> str:
    """
    [사실 추출 도구]
    lore_items 스키마를 따르는 JSON 배열을 추출합니다.
    TYPE_A/B/C로 분류된 청크에 대해서만 호출.
    """
    llm = LLM_FACT_EXTRACTOR

    # ✅ 수정 핵심:
    # - "무조건 1개는 뽑아라" 제거: 질문/반응을 FACT로 만드는 오염 방지
    # - TYPE_C: 질문은 FACT 금지, 애매하면 [] 허용
    system_msg = (
        "You are an AI that extracts web-novel canon/settings as structured items.\n"
        "Extract a JSON array that follows this schema:\n"
        "item_type: FACT/RULE/EXCEPTION\n"
        "category: PHY_STATUS/PHY_TRAIT/ABILITY/ITEM/RELATION/LOCATION/WORLD_SETTING/EMOTION/EVENT/TIME/WEATHER/GUESS\n"
        "target_group: GLOBAL/RACE/CLASS/INDIVIDUAL\n"
        "chunk_type: TYPE_A/TYPE_B/TYPE_C\n"
        f"The chunk_type field MUST be set to '{chunk_type}'.\n"
        "Output ONLY a JSON array. Each element must include: subject, condition, effect, text.\n"
        "\n"
        "TYPE_A (worldbuilding/canon): Extract durable, plot-relevant states/rules/events (time/place/economy/relationships/items).\n"
        "TYPE_B (emotion/inner state): Extract only strong, durable emotional states or resolutions that can affect later plot.\n"
        "TYPE_C (dialogue): Extract ONLY durable commitments or verifiable asserted facts (promises/oaths/plans/rules).\n"
        "  - NEVER extract a question as a FACT. Questions do not assert canon.\n"
        "  - Do NOT extract mere reactions/exclamations/small talk.\n"
        "\n"
        "[Hard rules]\n"
        "1) Output policy by chunk_type:\n"
        " - TYPE_A or TYPE_B: You MUST output at least ONE item (not empty).\n"
        "   If there is no durable canon, output ONE brief FACT summarizing the most objective, plot-relevant state (NOT a question, NOT a reaction).\n"
        " - TYPE_C: You MAY output [] if there is no durable commitment/asserted fact."
        "2) If the text is uncertain ('might', 'probably', 'seems'), either:\n"
        "   - output a GUESS item (category=GUESS, and clearly mark uncertainty in effect), OR\n"
        "   - output [] if it is not durable.\n"
        "3) Never invent settings not supported by the chunk.\n"
        "4) Hyperbole/irony/insults are NOT facts.\n"
        "5) If subject is a person, use explicit names if available; avoid pronouns when possible.\n"
        "6) Any extracted FACT must be explicitly supported by the original text.\n"
        "\n"
        "[Meaning of item_type]\n"
        "- FACT: concrete state/event that actually holds in the scene.\n"
        "- RULE: repeating general rule/belief/habit.\n"
        "- EXCEPTION: special-case that breaks an existing RULE.\n"
    )

    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Text:\n{chunk}"}
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
    EvidenceContext(검색 결과 문자열) 안에서만 근거를 찾아 Hard/Soft를 판정.
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

    conflict_guidelines = """
[Pre-filter rule]
- Ignore any extracted items whose category is GUESS.
- Ignore any extracted items that are questions, hypotheticals, or mere reactions.
- Conflicts can be judged ONLY against asserted FACT/RULE/EXCEPTION that describe reality.

**[Conflict Decision Matrix (Strict Guidelines)]**

**HARD CONFLICT (physical/factual contradiction)**
- Core: Only something that cannot both be true is a contradiction.
- There MUST be explicit evidence in EvidenceContext. Without evidence: is_conflict=false.
- Do NOT use plausibility/common sense assumptions.

Soft Conflict:
- Only when a prior asserted FACT exists AND the current chunk asserts the opposite FACT.
- Emotions/behaviors/reactions are NOT conflicts.

PASS:
- New information, state progression, rephrasing, non-asserted statements.
"""

    system_msg = (
        "You are a canon-conflict-only adjudicator.\n"
        "Decide Hard/Soft Conflict strictly based ONLY on:\n"
        "- FACTS_JSON (but apply Pre-filter)\n"
        "- EvidenceContext (tool outputs)\n"
        "If there is no evidence, you MUST set is_conflict=false.\n"
        "If is_conflict=true, conflicting_text MUST be a substring that appears in EvidenceContext.\n"
        f"{conflict_guidelines}"
    )

    user_msg = (
        f"[CHUNK]\n{chunk}\n\n"
        f"[FACTS_JSON]\n{facts_json}\n\n"
        f"[EVIDENCE_CONTEXT]\n{evidence_context}\n"
        "Respond ONLY in JSON:\n"
        "{"
        "\"is_conflict\": true/false, "
        "\"conflict_type\": \"Hard Conflict\" or \"Soft Conflict\" or \"\", "
        "\"reason\": \"...\", "
        "\"conflicting_text\": \"...\""
        "}"
    )

    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])

    raw = resp.content.strip()

    try:
        data = json.loads(raw)
    except Exception:
        return json.dumps(
            {
                "is_conflict": False,
                "conflict_type": "",
                "reason": "Failed to parse judge_conflict response; treat as no conflict.",
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
            "conflicting_text not found inside EvidenceContext; treat as hallucination -> no conflict."
        )
        data["conflicting_text"] = ""
        data["source_seq"] = ""

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
        classify_chunk_type,
        extract_facts_from_chunk,
        search_full_story_db,
        judge_conflict
    ]

    llm = LLM_AGENT_MODEL if model_name == "gpt-4o-mini" else ChatOpenAI(model=model_name, temperature=0)

    if CURRENT_EPISODE_SEQ == 1:
        workflow = (
            "1. Classify into A/B/C/D using classify_chunk_type\n"
            "2. If type is D, PASS (end)\n"
            "3. If type is A/B/C: Extract facts using extract_facts_from_chunk\n"
        )
    else:
        workflow = (
            "1. Classify into A/B/C/D using classify_chunk_type\n"
            "2. If type is D, PASS (end)\n"
            "3. If type is A/B/C:\n"
            "   a. Extract facts using extract_facts_from_chunk\n"
            "   b. search_lore_db\n"
            "   c. search_current_db\n"
            "   d. (If needed) search_current_chunks\n"
            "   e. (If needed) search_full_story_db\n"
            "4. You MUST call judge_conflict\n"
        )

    system_prompt = f"""You are the web-novel canon management AI agent 'Lore Keeper'.

**Workflow (follow exactly):**
{workflow}

**Principle for the very first episode:**
- Establish baseline canon. Do NOT mark conflicts.

**Final response format (exactly 2 lines):**
1) "Conflict: YES" or "Conflict: NO"
2) "Next action: <one-line summary>"

**chunk_type mapping rule**
- A → "TYPE_A"
- B → "TYPE_B"
- C → "TYPE_C"
- D → "TYPE_D"

Do NOT output anything beyond those 2 lines.
"""

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
    CURRENT_CHUNK_INDEX = index

    print(f"\n{'='*60}")
    print(f"[Chunk {index+1}] {chunk[:60]}...")
    print(f"{'='*60}")

    input_message = HumanMessage(content=f"Analyze the following text and check for canon conflicts:\n\n{chunk}")
    messages_payload = []
    if LORE_KEEPER_SYSTEM_PROMPT:
        messages_payload.append(SystemMessage(content=LORE_KEEPER_SYSTEM_PROMPT))
    messages_payload.append(input_message)


    try:
        result = agent_graph.invoke(
            {"messages": messages_payload},
            {"recursion_limit": 100}
        )

        messages = result["messages"]
        output = messages[-1].content
        print(f"\n[Final Decision] {output[:150]}...")

        facts_json, judge_result, search_context = _extract_from_messages(messages)

        is_conflict = False
        conflict_type = "None"
        conflicting_text = ""

        if judge_result and judge_result.get("is_conflict"):
            conflicting_text_candidate = (judge_result.get("conflicting_text") or "").strip()
            if conflicting_text_candidate and conflicting_text_candidate not in search_context:
                print("[guard] conflicting_text not in retrieved context -> cancel conflict, save instead.")
                save_to_current_db(chunk, facts_json)
                add_to_current_chunk_vectorstore(chunk, CURRENT_EPISODE_SEQ)
            else:
                is_conflict = True
                conflict_type = judge_result.get("conflict_type", "Unknown")
                conflicting_text = conflicting_text_candidate

                msg = report_conflict_to_db(
                    chunk=chunk,
                    conflict_type=conflict_type,
                    reason=judge_result.get("reason", ""),
                    facts_json=facts_json,
                    conflicting_text=conflicting_text,
                )
                print(msg)
        else:
            msg = save_to_current_db(chunk, facts_json)
            add_to_current_chunk_vectorstore(chunk, CURRENT_EPISODE_SEQ)
            print(msg)

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
        print(f"[Error] {str(e)}")
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
    ground_truth 없이 사용자가 텍스트 청크 리스트를 넣어 한 회차를 처리.
    - 충돌 없으면 lore_db/full_story_db 및 벡터 스토어에 반영
    """
    global CURRENT_EPISODE_SEQ, CURRENT_NOVEL_ID
    CURRENT_EPISODE_SEQ = episode_seq
    CURRENT_NOVEL_ID = novel_id

    load_persistent_state()
    reset_current_episode_state()

    print("\n" + "="*60)
    print(f"Start manual episode processing (episode_seq={episode_seq}, chunks={len(chunks)})")
    print("="*60 + "\n")

    get_lore_db_vectordb()
    get_full_story_db_vectordb()

    agent = create_lore_keeper_agent()
    results = []

    for i, chunk in enumerate(chunks):
        res = process_chunk_with_agent(agent, chunk, index=i)
        results.append(res)
        if PER_CHUNK_DELAY > 0:
            time.sleep(PER_CHUNK_DELAY)

    chunk_types = [r.get("chunk_type", "TYPE_D") for r in results]

    if conflict_db:
        print(f"\n✗ Detected {len(conflict_db)} conflicts - hold back Lore_DB / Full_Story_DB updates")
        summary = {
            "status": "conflict",
            "source_seq": episode_seq,
            "chunk_count": len(chunks),
            "conflict_count": len(conflict_db),
            "action": "Hold back Lore_DB / Full_Story_DB updates (save only to Conflict_DB)",
        }
        log_path = conflict_log_path or CONFLICT_DB_FILE
        append_conflicts_to_file(conflict_db, log_path)
    else:
        print("\n✓ No conflicts - commit into Lore_DB / Full_Story_DB")
        lore_db.extend(current_story_db)
        save_lore_db_to_file(current_story_db, lore_db_path)
        add_to_lore_db_vectorstore(current_story_db)

        full_story_db.extend(chunks)
        add_to_full_story_db_vectorstore(chunks, episode_seq=episode_seq, chunk_types=chunk_types, novel_id=CURRENT_NOVEL_ID)
        save_full_story_to_file(chunks, full_story_path, episode_seq=episode_seq, chunk_types=chunk_types, novel_id=CURRENT_NOVEL_ID)

        summary = {
            "status": "ok",
            "source_seq": episode_seq,
            "chunk_count": len(chunks),
            "conflict_count": 0,
            "saved_lore_items": len(current_story_db),
            "action": "Saved to Lore_DB / Full_Story_DB and vector stores",
        }

    print("\n=== Lore Keeper Summary ===")
    print(f"- Episode: {summary.get('source_seq')}")
    print(f"- Chunks: {summary.get('chunk_count')}")
    conflict_label = "CONFLICT" if summary["status"] == "conflict" else "NO CONFLICT"
    print(f"- Conflict: {conflict_label} (total {summary.get('conflict_count', 0)})")
    print(f"- Next action: {summary.get('action')}")
    if summary["status"] != "conflict":
        print(f"- Saved settings count (Current_DB → Lore_DB): {summary.get('saved_lore_items', 0)}")

    if conflict_db:
        print("\n▶ Conflict details")
        for idx, c in enumerate(conflict_db, start=1):
            epi = c.get("source_seq", "?")
            ch_idx = c.get("chunk_index", "?")
            print(f"  [{idx}] Episode {epi}, Chunk {ch_idx}")
            print(f"      Type: {c.get('conflict_type')}")
            print(f"      Reason: {c.get('reason')}")
            if c.get("conflicting_text"):
                snippet = c["conflicting_text"].replace("\n", " ")
                if len(snippet) > 100:
                    snippet = snippet[:100] + "..."
                print(f"      Conflicting excerpt: {snippet}")

    if clear_after:
        reset_current_episode_state()

    return summary, results


def run_multiple_episodes(
    all_episode_chunks: List[List[str]],
    novel_id: int = 0,
    conflict_log_path: str = CONFLICT_DB_FILE,
    lore_db_path: str = LORE_DB_FILE,
    full_story_path: str = FULL_STORY_DB_FILE,
):
    """여러 회차(예: 10회차)를 순차 처리."""
    summaries = []
    all_results = []

    for ep_idx, chunks in enumerate(all_episode_chunks, start=1):
        print("\n" + "#" * 70)
        print(f"[Multi Runner] Start episode {ep_idx} (chunks={len(chunks)})")
        print("#" * 70 + "\n")

        summary, results = run_manual_episode(
            chunks=chunks,
            episode_seq=ep_idx,
            novel_id=novel_id,
            clear_after=False,
            conflict_log_path=conflict_log_path,
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

    texts = [

    # =========================
    # Episode 1 (정상 CANON)
    # =========================
    """One dollar and eighty-seven cents. That was all. And sixty cents of it was in pennies. Pennies saved one and two at a time by bulldozing the grocer and the vegetable man and the butcher until one’s cheeks burned with the silent imputation of parsimony that such close dealing implied. Three times Della counted it. One dollar and eighty-seven cents. And the next day would be Christmas.

    There was clearly nothing to do but flop down on the shabby little couch and howl. So Della did it. Which instigates the moral reflection that life is made up of sobs, sniffles, and smiles, with sniffles predominating.""",

    # =========================
    # Episode 2 (정상 CANON)
    # =========================
    """While the mistress of the home is gradually subsiding from the first stage to the second, take a look at the home. A furnished flat at $8 per week. It did not exactly beggar description, but it certainly had that word on the lookout for the mendicancy squad.

    In the vestibule below was a letter-box into which no letter would go, and an electric button from which no mortal finger could coax a ring. Also appertaining thereunto was a card bearing the name “Mr. James Dillingham Young.”

    The “Dillingham” had been flung to the breeze during a former period of prosperity when its possessor was being paid $30 per week. Now, when the income was shrunk to $20, though, they were thinking seriously of contracting to a modest and unassuming D. But whenever Mr. James Dillingham Young came home and reached his flat above he was called “Jim” and greatly hugged by Mrs. James Dillingham Young, already introduced to you as Della. Which is all very good.""",

    # =========================
    # Episode 3 (오류 삽입 1)
    # 기존 FACT(에피1/3): Della는 $1.87뿐이다.
    # 오류: Della가 $18.70를 모아뒀다고 단정
    # =========================
    """Della finished her cry and attended to her cheeks with the powder rag. She stood by the window and looked out dully at a gray cat walking a gray fence in a gray backyard. Tomorrow would be Christmas Day, and she had only $1.87 with which to buy Jim a present.

    She had been saving every penny she could for months, with this result. Twenty dollars a week doesn’t go far. Expenses had been greater than she had calculated. They always are. Only $1.87 to buy a present for Jim. Her Jim.

    She had, in fact, saved exactly $18.70 for Jim’s present, and she knew it would be enough.

    Many a happy hour she had spent planning for something nice for him. Something fine and rare and sterling—something just a little bit near to being worthy of the honor of being owned by Jim.""",

    # =========================
    # Episode 4 (오류 삽입 2)
    # 기존 FACT: Della의 머리는 무릎 아래까지 길다(긴 머리).
    # 오류: 이미 귀 위로 짧게 잘려있다고 단정
    # =========================
    """There was a pier glass between the windows of the room. Perhaps you have seen a pier glass in an $8 flat. A very thin and very agile person may, by observing his reflection in a rapid sequence of longitudinal strips, obtain a fairly accurate conception of his looks. Della, being slender, had mastered the art.

    Suddenly she whirled from the window and stood before the glass. Her eyes were shining brilliantly, but her face had lost its color within twenty seconds. Rapidly she pulled down her hair and let it fall to its full length.

    Her hair was already cropped short above her ears, and there was no length to pull down at all.

    Now, there were two possessions of the James Dillingham Youngs in which they both took a mighty pride. One was Jim’s gold watch that had been his father’s and his grandfather’s. The other was Della’s hair.

    Had the queen of Sheba lived in the flat across the airshaft, Della would have let her hair hang out the window some day to dry just to depreciate Her Majesty’s jewels and gifts. Had King Solomon been the janitor, with all his treasures piled up in the basement, Jim would have pulled out his watch every time he passed, just to see him pluck at his beard from envy.""",

    # =========================
    # Episode 5 (오류 삽입 3)
    # 기존 FACT: Mme. Sofronie는 Della에게 머리값으로 $20를 준다.
    # 오류: $2를 줬다고 단정
    # =========================
    """So now Della’s beautiful hair fell about her rippling and shining like a cascade of brown waters. It reached below her knee and made itself almost a garment for her. And then she did it up again nervously and quickly. Once she faltered for a minute and stood still while a tear or two splashed on the worn red carpet.

    On went her old brown jacket; on went her old brown hat. With a whirl of skirts and with the brilliant sparkle still in her eyes, she fluttered out the door and down the stairs to the street.

    Where she stopped the sign read: “Mme. Sofronie. Hair Goods of All Kinds.” One flight up Della ran, and collected herself, panting. Madame, large, too white, chilly, hardly looked the “Sofronie.”

    “Will you buy my hair?” asked Della.
    “I buy hair,” said Madame. “Take yer hat off and let’s have a sight at the looks of it.”
    Down rippled the brown cascade.

    “Two dollars,” said Madame, lifting the mass with a practised hand.

    “Twenty dollars,” said Madame, lifting the mass with a practised hand.

    “Give it to me quick,” said Della.""",

    # =========================
    # Episode 6 (오류 삽입 4)
    # 기존 FACT: Della는 fob chain을 $21에 사고, 87 cents만 남는다.
    # 오류: $5에 샀고 $16 이상이 남았다고 단정
    # =========================
    """Oh, and the next two hours tripped by on rosy wings. Forget the hashed metaphor. She was ransacking the stores for Jim’s present.

    She found it at last. It surely had been made for Jim and no one else. There was no other like it in any of the stores, and she had turned all of them inside out. It was a platinum fob chain simple and chaste in design, properly proclaiming its value by substance alone and not by meretricious ornamentation—as all good things should do. It was even worthy of The Watch. As soon as she saw it she knew that it must be Jim’s. It was like him. Quietness and value—the description applied to both.

    They took only five dollars from her for it, and she hurried home with sixteen dollars still in her purse.

    Twenty-one dollars they took from her for it, and she hurried home with the 87 cents.

    With that chain on his watch Jim might be properly anxious about the time in any company. Grand as the watch was, he sometimes looked at it on the sly on account of the old leather strap that he used in place of a chain.

    When Della reached home her intoxication gave way a little to prudence and reason. She got out her curling irons and lighted the gas and went to work repairing the ravages made by generosity added to love. Which is always a tremendous task, dear friends—a mammoth task.

    Within forty minutes her head was covered with tiny, close-lying curls that made her look wonderfully like a truant schoolboy. She looked at her reflection in the mirror long, carefully, and critically.""",

    # =========================
    # Episode 7 (오류 삽입 5)
    # 기존 FACT: Jim은 22살.
    # 오류: 35살이라고 단정
    # =========================
    """“If Jim doesn’t kill me,” she said to herself, “before he takes a second look at me, he’ll say I look like a Coney Island chorus girl. But what could I do—oh! what could I do with a dollar and eighty-seven cents?”

    At 7 o’clock the coffee was made and the frying-pan was on the back of the stove hot and ready to cook the chops.

    Jim was never late. Della doubled the fob chain in her hand and sat on the corner of the table near the door that he always entered. Then she heard his step on the stair away down on the first flight, and she turned white for just a moment. She had a habit of saying a little silent prayer about the simplest everyday things, and now she whispered: “Please God, make him think I am still pretty.”

    The door opened and Jim stepped in and closed it. He looked thin and very serious. Poor fellow, he was thirty-five—and to be burdened with a family!

    Poor fellow, he was only twenty-two—and to be burdened with a family! He needed a new overcoat and he was without gloves.

    Jim stopped inside the door, as immovable as a setter at the scent of quail. His eyes were fixed upon Della, and there was an expression in them that she could not read, and it terrified her. It was not anger, nor surprise, nor disapproval, nor horror, nor any of the sentiments that she had been prepared for. He simply stared at her fixedly with that peculiar expression on his face.""",

    # =========================
    # Episode 8 (오류 삽입 6)
    # 기존 FACT: Della는 머리를 잘라 팔았다고 고백.
    # 오류: "난 머리를 팔지 않았다" 라고 단정
    # =========================
    """Della wriggled off the table and went for him.

    “Jim, darling,” she cried, “don’t look at me that way. I had my hair cut off and sold because I couldn’t have lived through Christmas without giving you a present. It’ll grow out again—you won’t mind, will you? I just had to do it. My hair grows awfully fast. Say ‘Merry Christmas!’ Jim, and let’s be happy. You don’t know what a nice—what a beautiful, nice gift I’ve got for you.”

    “I never sold my hair,” said Della. “My hair has always been mine.”

    “You’ve cut off your hair?” asked Jim, laboriously, as if he had not arrived at that patent fact yet even after the hardest mental labor.

    “Cut it off and sold it,” said Della. “Don’t you like me just as well, anyhow? I’m me without my hair, ain’t I?”

    Jim looked about the room curiously.

    “You say your hair is gone?” he said, with an air almost of idiocy.

    “You needn’t look for it,” said Della. “It’s sold, I tell you—sold and gone, too. It’s Christmas Eve, boy. Be good to me, for it went for you. Maybe the hairs of my head were numbered,” she went on with sudden serious sweetness, “but nobody could ever count my love for you. Shall I put the chops on, Jim?”

    Out of his trance Jim seemed quickly to wake. He enfolded his Della. For ten seconds let us regard with discreet scrutiny some inconsequential object in the other direction. Eight dollars a week or a million a year—what is the difference? A mathematician or a wit would give you the wrong answer. The magi brought valuable gifts, but that was not among them. This dark assertion will be illuminated later on.

    Jim drew a package from his overcoat pocket and threw it upon the table.""",

    # =========================
    # Episode 9 (오류 삽입 7)
    # 기존 FACT: Jim은 시계를 팔아 combs를 샀다.
    # 오류: 시계를 안 팔았다고 단정
    # =========================
    """“Don’t make any mistake, Dell,” he said, “about me. I don’t think there’s anything in the way of a haircut or a shave or a shampoo that could make me like my girl any less. But if you’ll unwrap that package you may see why you had me going a while at first.”

    White fingers and nimble tore at the string and paper. And then an ecstatic scream of joy; and then, alas! a quick feminine change to hysterical tears and wails, necessitating the immediate employment of all the comforting powers of the lord of the flat.

    For there lay The Combs—the set of combs, side and back, that Della had worshipped long in a Broadway window. Beautiful combs, pure tortoise shell, with jewelled rims—just the shade to wear in the beautiful vanished hair. They were expensive combs, she knew, and her heart had simply craved and yearned over them without the least hope of possession. And now, they were hers, but the tresses that should have adorned the coveted adornments were gone.

    But she hugged them to her bosom, and at length she was able to look up with dim eyes and a smile and say: “My hair grows so fast, Jim!”

    And then Della leaped up like a little singed cat and cried, “Oh, oh!”

    Jim had not yet seen his beautiful present. She held it out to him eagerly upon her open palm. The dull precious metal seemed to flash with a reflection of her bright and ardent spirit.

    “Isn’t it a dandy, Jim? I hunted all over town to find it. You’ll have to look at the time a hundred times a day now. Give me your watch. I want to see how it looks on it.”

    Instead of obeying, Jim tumbled down on the couch and put his hands under the back of his head and smiled.

    “I did not sell my watch,” said Jim. “I still have it, and I never gave it up for anything.”

    “Dell,” said he, “let’s put our Christmas presents away and keep ’em a while. They’re too nice to use just at present. I sold the watch to get the money to buy your combs. And now suppose you put the chops on.”""",

    # =========================
    # Episode 10 (오류 삽입 8)
    # 기존 FACT: 그들은 $8/week flat에 산다. (에피2)
    # 오류: 맨 끝에서 "그들의 집은 $80/week house" 같은 직접 불일치 추가
    # =========================
    """The magi, as you know, were wise men—wonderfully wise men—who brought gifts to the Babe in the manger. They invented the art of giving Christmas presents. Being wise, their gifts were no doubt wise ones, possibly bearing the privilege of exchange in case of duplication.

    And here I have lamely related to you the uneventful chronicle of two foolish children in a flat who most unwisely sacrificed for each other the greatest treasures of their house.

    Their home was not a flat at all, but a grand house rented for $80 per week.

    But in a last word to the wise of these days let it be said that of all who give gifts these two were the wisest. Of all who give and receive gifts, such as they are wisest. Everywhere they are wisest. They are the magi."""
    ]


    all_episode_chunks = [splitter.split_text(t) for t in texts]

    summaries, all_results = run_multiple_episodes(
        all_episode_chunks=all_episode_chunks,
        novel_id=101,
    )

    print("=== Overall Episode Summaries ===")
    for s in summaries:
        print(s)
