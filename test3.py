import os
import json
import time
from typing import List, Dict, Any, Optional, Set

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

# 전역 시스템 프롬프트 캐시
LORE_KEEPER_SYSTEM_PROMPT = ""

# ========= 경로 설정 =========
LORE_DB_VECTOR_DIR = "./chroma_lore_db"  # lore_db: 정규화된 lore_items (설정 검색용)
FULL_STORY_DB_VECTOR_DIR = "./chroma_full_story_db"  # Full_Story_DB: 원본 chunk_data (맥락 검색용)
LORE_DB_FILE = "./lore_db.csv"
FULL_STORY_DB_FILE = "./full_story_db.csv"
CONFLICT_DB_FILE = "./conflict_db.csv"

# ========= 실행 설정 =========
DEFAULT_EPISODE_SEQ = 1

# ========= OpenAI API Key 설정 =========

# ========= 전역 변수 =========
current_story_db: List[Dict[str, Any]] = []  # 현재 회차 lore_items (RDBMS 역할)
conflict_db: List[Dict[str, Any]] = []  # 충돌 내역
lore_db: List[Dict[str, Any]] = []  # 회차 확정 후 정규화된 lore_items 누적 (RDBMS)
full_story_db: List[str] = []  # 회차 확정 후 원본 chunk_data 누적 (RDBMS)
_lore_db_vectordb: Optional[Chroma] = None  # lore_db 벡터 스토어 (정규화된 설정)
_full_story_db_vectordb: Optional[Chroma] = None  # Full_Story_DB 벡터 스토어 (원본 청크)
_current_story_vectordb: Optional[Chroma] = None  # current_story_db 임시 벡터 스토어
_current_chunk_vectordb: Optional[Chroma] = None  # 현재 회차 원문 청크 임시 벡터 스토어
_state_loaded: bool = False

# Embeddings 객체 (재사용)
embeddings = OpenAIEmbeddings()



# ========= DB 스키마 =========
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


# ========= 벡터 스토어 관리 =========


def get_lore_db_vectordb() -> Chroma:
    """lore_db 전용 벡터 스토어 생성/로드 (정규화된 lore_items)"""
    global _lore_db_vectordb

    if _lore_db_vectordb is not None:
        return _lore_db_vectordb

    if os.path.exists(LORE_DB_VECTOR_DIR):
        print(f"[Lore_DB_VectorStore] 기존 벡터 스토어 로드: {LORE_DB_VECTOR_DIR}")
        _lore_db_vectordb = Chroma(
            persist_directory=LORE_DB_VECTOR_DIR,
            embedding_function=embeddings,
            collection_name="lore_db",
        )
    else:
        print(f"[Lore_DB_VectorStore] 새 벡터 스토어 생성")
        _lore_db_vectordb = Chroma(
            persist_directory=LORE_DB_VECTOR_DIR,
            embedding_function=embeddings,
            collection_name="lore_db",
        )

    return _lore_db_vectordb


def get_full_story_db_vectordb() -> Chroma:
    """Full_Story_DB 전용 벡터 스토어 생성/로드 (원본 chunk_data)"""
    global _full_story_db_vectordb

    if _full_story_db_vectordb is not None:
        return _full_story_db_vectordb

    if os.path.exists(FULL_STORY_DB_VECTOR_DIR):
        print(f"[Full_Story_DB_VectorStore] 기존 벡터 스토어 로드: {FULL_STORY_DB_VECTOR_DIR}")
        _full_story_db_vectordb = Chroma(
            persist_directory=FULL_STORY_DB_VECTOR_DIR,
            embedding_function=embeddings,
            collection_name="full_story_db",
        )
    else:
        print(f"[Full_Story_DB_VectorStore] 새 벡터 스토어 생성")
        _full_story_db_vectordb = Chroma(
            persist_directory=FULL_STORY_DB_VECTOR_DIR,
            embedding_function=embeddings,
            collection_name="full_story_db",
        )

    return _full_story_db_vectordb


def add_to_lore_db_vectorstore(lore_items: List[Dict[str, Any]]):
    """Current_DB의 정규화된 lore_items를 lore_db 벡터 스토어에 추가"""
    if not lore_items:
        return

    vectordb = get_lore_db_vectordb()

    texts = []
    metadatas = []

    for item in lore_items:
        # 검색 가능한 텍스트 생성 (설정 내용 중심)
        searchable_text = (
            f"[{item['item_type']}/{item['category']}] "
            f"Subject: {item['subject']}, "
            f"Effect: {item['effect']}, "
            f"Condition: {item['condition']}, "
            f"Text: {item['text']}"
        )
        texts.append(searchable_text)

        # 메타데이터에 전체 정보 저장
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

    # 벡터 스토어에 추가
    vectordb.add_texts(texts=texts, metadatas=metadatas)
    print(f"[Lore_DB_VectorStore] {len(texts)}개의 lore_items가 벡터화되어 저장되었습니다.")


def add_to_full_story_db_vectorstore(chunk_data_list: List[str]):
    """원본 chunk_data를 Full_Story_DB 벡터 스토어에 추가"""
    if not chunk_data_list:
        return

    vectordb = get_full_story_db_vectordb()

    # 원본 청크를 그대로 벡터화
    metadatas = [{"chunk_index": i, "source": "processed_episode"} for i in range(len(chunk_data_list))]

    vectordb.add_texts(texts=chunk_data_list, metadatas=metadatas)
    print(f"[Full_Story_DB_VectorStore] {len(chunk_data_list)}개의 chunk_data가 벡터화되어 저장되었습니다.")


def get_current_chunk_vectordb() -> Chroma:
    """현재 회차 원문 청크 전용 임시 벡터 스토어"""
    
    global _current_chunk_vectordb
    if _current_chunk_vectordb is not None:
        return _current_chunk_vectordb

    _current_chunk_vectordb = Chroma(
        collection_name="current_chunks_tmp",
        embedding_function=embeddings,
    )
    print(f"current 벡터 스토어 생성")
    return _current_chunk_vectordb


def add_to_current_chunk_vectorstore(chunks: List[str], episode_seq: int = DEFAULT_EPISODE_SEQ):
    """현재 회차 원문 청크를 임시 벡터 스토어에 추가"""
    if not chunks:
        return
    print(f"add_to_current_chunk_vectorstore: {len(chunks)}")
    vectordb = get_current_chunk_vectordb()
    print(f"vectordb: {vectordb}")
    metadatas = [{"chunk_index": i, "source_seq": str(episode_seq)} for i in range(len(chunks))]
    # print(f"metadatas: {metadatas}")
    # try:
    #     vectordb.add_texts(texts=chunks, metadatas=metadatas)
    #     print(f"[Current_Chunk_VectorStore] {len(chunks)}개의 원문 청크가 임시 벡터에 추가되었습니다.")
    # except Exception as e:
    #     print(f"Error: {e}")
    print(f"metadatas 생성 완료: 총 {len(metadatas)}개")

    # [수정] 5개씩 끊어서 저장하며 생사 확인 (Batch Processing)
    batch_size = 5
    total_chunks = len(chunks)

    print(f"[Current_Chunk_VectorStore] 총 {total_chunks}개 저장 시작...")

    try:
        for i in range(0, total_chunks, batch_size):
            # 1. 배치 자르기
            end_idx = min(i + batch_size, total_chunks)
            batch_texts = chunks[i : end_idx]
            batch_metadatas = metadatas[i : end_idx]
            
            print(f"   -> [{i}~{end_idx-1}]번 청크 저장 시도 중...", end="", flush=True) # 강제 출력
            
            # 2. 저장 실행
            vectordb.add_texts(texts=batch_texts, metadatas=batch_metadatas)
            
            print(" 성공!") # 이 메시지가 안 나오면 여기서 죽은 것
            time.sleep(0.2) # 잠깐 숨 고르기

        print(f"[Current_Chunk_VectorStore] 모든 청크 저장 완료!")

    except Exception as e:
        print(f"\n[Error] 저장 중 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def get_current_story_vectordb() -> Chroma:
    """current_story_db 전용 벡터 스토어 생성/로드 (임시, non-persist)"""
    global _current_story_vectordb
    if _current_story_vectordb is not None:
        return _current_story_vectordb
    _current_story_vectordb = Chroma(
        collection_name="current_story_tmp",
        embedding_function=embeddings,
    )
    return _current_story_vectordb


def add_to_current_db_vectorstore(lore_items: List[Dict[str, Any]]):
    """Current_DB에 저장된 lore_items를 임시 벡터 스토어에 추가"""
    if not lore_items:
        return

    vectordb = get_current_story_vectordb()
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

    vectordb.add_texts(texts=texts, metadatas=metadatas)
    print(f"[Current_DB_VectorStore] {len(texts)}개의 lore_items가 임시 벡터에 추가되었습니다.")


def reset_current_episode_state():
    """현재 회차 상태 초기화 (current/ conflict/ 임시 벡터)"""
    current_story_db.clear()
    conflict_db.clear()
    global _current_story_vectordb
    _current_story_vectordb = None
    global _current_chunk_vectordb
    _current_chunk_vectordb = None


def append_conflicts_to_file(conflicts: List[Dict[str, Any]], path: str):
    """conflict_db 내용을 파일에 누적 저장 (CSV)"""
    if not conflicts:
        return
    df_new = pd.DataFrame(conflicts)
    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            pass
    df_new.to_csv(path, index=False)
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
    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            pass
    df_new.to_csv(path, index=False)
    print(f"[Lore_DB_File] {len(items)}개 저장/누적 → {path}")


def save_full_story_to_file(chunks: List[str], path: str = FULL_STORY_DB_FILE, episode_seq: int = DEFAULT_EPISODE_SEQ):
    if not chunks:
        return
    df_new = pd.DataFrame(
        [{"text": ck, "source_seq": episode_seq, "chunk_index": i} for i, ck in enumerate(chunks)]
    )
    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            pass
    df_new.to_csv(path, index=False)
    print(f"[Full_Story_File] {len(chunks)}개 저장/누적 → {path}")


def load_persistent_state():
    """CSV에서 lore_db/full_story_db를 로드하고 벡터에도 반영"""
    global _state_loaded
    if _state_loaded:
        return

    if os.path.exists(LORE_DB_FILE):
        try:
            df = pd.read_csv(LORE_DB_FILE)
            for _, row in df.iterrows():
                metadata = {}
                if isinstance(row.get("metadata"), str):
                    try:
                        metadata = json.loads(row["metadata"])
                    except Exception:
                        metadata = {}
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
            if lore_db:
                add_to_lore_db_vectorstore(lore_db)
                print(f"[Load] lore_db {len(lore_db)}개 로드 및 벡터 반영")
        except Exception as e:
            print(f"[Load] lore_db 로드 실패: {e}")

    if os.path.exists(FULL_STORY_DB_FILE):
        try:
            df = pd.read_csv(FULL_STORY_DB_FILE)
            texts = df["text"].tolist()
            full_story_db.extend(texts)
            if texts:
                add_to_full_story_db_vectorstore(texts)
                print(f"[Load] full_story_db {len(texts)}개 로드 및 벡터 반영")
        except Exception as e:
            print(f"[Load] full_story_db 로드 실패: {e}")

    _state_loaded = True


# ========= Tools 정의 =========

@tool
def search_current_db(query: str) -> str:
    """
    [Current_DB 검색 도구]
    현재 회차에서 정규화된 설정을 임시 벡터 스토어에서 검색합니다.
    """
    if not current_story_db:
        return "[Current_DB가 비어있습니다]"

    vectordb = get_current_story_vectordb()
    docs = vectordb.similarity_search(query, k=5)

    if not docs:
        return "[Current_DB에서 관련 설정 없음]"

    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        lines.append(
            f"{i}. [{meta.get('item_type')}/{meta.get('category')}] "
            f"{meta.get('subject')} -> {meta.get('effect')}\n"
            f"   원문: {meta.get('text', '')}"
        )

    return "[Current_DB 검색 결과 (임시 설정)]\n" + "\n\n".join(lines)

# 원본을 직접 검색하지는 않음. -> 그럴 용도로 full_story_db_vectordb를 사용함.
# @tool
# def search_story_context(query: str) -> str:
#     """
#     [스토리 맥락 검색 도구]
#     원본 story.txt VectorStore에서 관련 맥락을 검색합니다.
#     """
#     vectordb = build_story_vectordb()
#     docs = vectordb.similarity_search(query, k=5)
#     context = "\n\n".join([d.page_content for d in docs])
#     return f"[스토리 맥락 검색 결과]\n{context}" if context else "[관련 스토리 맥락 없음]"


@tool
def search_lore_db(query: str) -> str:
    """
    [Lore_DB 검색 도구]
    이전 회차들에서 확정된 설정들(정규화된 lore_items)을 벡터 검색합니다.
    """
    if not lore_db:
        return "[Lore_DB가 비어있습니다]"

    vectordb = get_lore_db_vectordb()
    docs = vectordb.similarity_search(query, k=5)

    if not docs:
        return "[Lore_DB에서 관련 설정 없음]"

    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        lines.append(
            f"{i}. [{meta.get('item_type')}/{meta.get('category')}] "
            f"{meta.get('subject')} -> {meta.get('effect')}\n"
            f"   원문: {meta.get('text', '')}"
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
    docs = vectordb.similarity_search(query, k=5)

    if not docs:
        return "[Full_Story_DB에서 관련 맥락 없음]"

    context = "\n\n".join([f"[청크 {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
    return "[Full_Story_DB 검색 결과 (원본 맥락)]\n" + context


@tool
def search_current_chunks(query: str) -> str:
    """
    [Current_Chunks 검색 도구]
    현재 회차의 원문 청크를 임시 벡터 스토어에서 검색합니다.
    """
    if not _current_chunk_vectordb:
        return "[Current_Chunks가 비어있습니다]"

    vectordb = get_current_chunk_vectordb()
    docs = vectordb.similarity_search(query, k=5)

    if not docs:
        return "[Current_Chunks에서 관련 맥락 없음]"

    lines = []
    for i, doc in enumerate(docs, 1):
        lines.append(f"{i}. {doc.page_content}")
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
            f"  원문: {item['text']}"
        )

    return "[Current_DB 설정들]\n" + "\n".join(lines)


@tool
def classify_chunk_type(chunk: str) -> str:
    """
    [청크 타입 분류 도구]
    A(설정), B(감정), C(대화), D(단순 서술)로 분류합니다.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
    TYPE_D면 빈 배열 반환.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_msg = (
        "너는 웹소설 설정 추출 AI야.\n"
        "주어진 문장에서 lore_items 스키마를 추출해:\n"
        "item_type: FACT/RULE/EXCEPTION\n"
        "category: PHY_STATUS/PHY_TRAIT/ABILITY/ITEM/RELATION/LOCATION/WORLD_SETTING\n"
        "target_group: GLOBAL/RACE/CLASS/INDIVIDUAL\n"
        "chunk_type: TYPE_A/TYPE_B/TYPE_C/TYPE_D\n"
        "subject, condition, effect, text를 포함한 JSON 배열만 출력.\n"
        "TYPE_D면 []만 출력."
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
        normalized = [_build_lore_item(fact, chunk) for fact in facts]
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
    """
    try:
        facts = json.loads(facts_json) if facts_json else []
        normalized_type = "Hard Conflict" if "hard" in conflict_type.lower() else "Soft Conflict"

        entry = {
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
            msg += f"\n  기존 설정: {conflicting_text[:50]}..."
        return msg
    except Exception as e:
        return f"✗ 저장 실패: {str(e)}"


# ========= 에이전트 생성 =========

def create_lore_keeper_agent(model_name: str = "gpt-4o-mini"):
    """Lore Keeper 에이전트 생성"""
    tools = [
        # search_story_context,
        search_lore_db,
        search_current_db,
        search_current_chunks,
        # search_full_story_db,
        get_current_db_settings,
        classify_chunk_type,
        extract_facts_from_chunk,
        save_to_current_db,
        report_conflict_to_db,
    ]

    llm = ChatOpenAI(model=model_name, temperature=0)

    workflow = (
        "1. classify_chunk_type으로 A/B/C/D 분류\n"
        "2. D 타입이면 PASS (작업 종료)\n"
        "3. A/B/C 타입이면:\n"
        "   a. extract_facts_from_chunk로 사실 추출\n"
        # "   b. search_story_context (원본 story.txt 맥락 검색)\n"
        "   c. search_lore_db (이전 회차 정규화된 설정 검색)\n"
        "   d. search_current_db (현재 회차 임시 설정 검색)\n"
        "   e. search_current_chunks (현재 회차 원문 맥락 검색)\n"
        # "   f. search_full_story_db (이전 회차 원본 맥락 검색)\n"
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

**중요:**
- 각 단계마다 필요한 tool을 스스로 선택해서 호출
- search_lore_db: 정규화된 설정 비교용
- search_current_db: 현재 회차 임시 설정 비교용
- search_current_chunks: 현재 회차 원문 맥락 확인용
- search_full_story_db: 원본 맥락 확인용 (필요할 때만)
- report_conflict_to_db 호출 시 conflicting_text에 충돌 원문 반드시 포함
- 판단 근거를 명확히 설명

지금부터 주어진 문장을 분석해봐."""

    global LORE_KEEPER_SYSTEM_PROMPT
    LORE_KEEPER_SYSTEM_PROMPT = system_prompt

    agent_graph = create_react_agent(
        model=llm,
        tools=tools,
    )
    return agent_graph


# ========= 청크 처리 =========

def process_chunk_with_agent(agent_graph, chunk: str, index: int) -> Dict[str, Any]:
    """에이전트로 청크 하나 처리"""
    print(f"\n{'='*60}")
    print(f"[Chunk {index+1}] {chunk[:60]}...")
    print(f"{'='*60}")

    input_message = HumanMessage(content=f"다음 문장을 분석해서 설정 충돌을 검사해줘:\n\n{chunk}")
    messages_payload = []
    if LORE_KEEPER_SYSTEM_PROMPT:
        messages_payload.append(SystemMessage(content=LORE_KEEPER_SYSTEM_PROMPT))
    messages_payload.append(input_message)

    # 현재 회차 원문 청크를 임시 벡터 스토어에 추가 (검색용)
    add_to_current_chunk_vectorstore([chunk])

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


# ========= 평가 메트릭 =========

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
    - 충돌 없으면 lore_db/full_story_db 및 각 벡터 스토어에 반영
    """
    load_persistent_state()
    reset_current_episode_state()

    print("\n" + "="*60)
    print(f"수동 회차 처리 시작 (episode_seq={episode_seq}, chunks={len(chunks)})")
    print("="*60 + "\n")

    # 벡터 스토어 준비 (lore/full 검색용)
    get_lore_db_vectordb()
    get_full_story_db_vectordb()
    add_to_current_chunk_vectorstore(chunks, episode_seq=episode_seq)

    agent = create_lore_keeper_agent()
    results = []

    for i, chunk in enumerate(chunks):
        res = process_chunk_with_agent(agent, chunk, index=i)
        results.append(res)
        time.sleep(0.2)

    if conflict_db:
        print(f"\n✗ 충돌 {len(conflict_db)}건 감지 - DB 반영 보류")
        summary = {"status": "conflict", "conflicts": conflict_db}
        log_path = conflict_log_path or CONFLICT_DB_FILE
        append_conflicts_to_file(conflict_db, log_path)
    else:
        print("\n✓ 충돌 없음 - lore_db / full_story_db에 반영")
        lore_db.extend(current_story_db)
        add_to_lore_db_vectorstore(current_story_db)
        save_lore_db_to_file(current_story_db, lore_db_path)
        full_story_db.extend(chunks)
        add_to_full_story_db_vectorstore(chunks)
        save_full_story_to_file(chunks, full_story_path, episode_seq=episode_seq)
        summary = {"status": "ok", "saved": len(current_story_db)}

    if clear_after:
        reset_current_episode_state()

    return summary, results


# ========= main =========

import re

if __name__ == "__main__":
    # 예시: 사용자가 텍스트 문자열을 직접 입력해 한 회차를 처리하려면 아래를 수정/사용하세요.
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
 처음에는 늦게 들어오는 사람들 때문에 수성수성하던 장내가 인제는 기침 소리 하나 없이 조용해졌다. 사회자는 말을 이어,
 "긴 말씀은 허지 않겠으나, 차나 마셔 가면서 간담적으로 피차에 의견도 교환하고, 그 동안에 분투한 체험담도 들려 주셔서 앞으로 이 운동을 계속하는 데 크게 참고가 되게 해주시 기를 바라는 바입니다."
 라고 부탁을 한 후 단에서 내렸다.
 대원들 중에서 제일 나이가 들어 보이는 어느 전문학교의 교복을 입은 학생이 나아가 간단한 답사를 하고 돌아왔다.
 문간에서 회장을 정돈시키던 이 신문사의 배지를 붙인 사원이 눈짓을 하니까, L여학교 가사과의 학생들은, 굉장한 연회나 차리는 듯이 일제히 에이프런을 두르고 돌아다니며 자기네의 손으로 만든 과자와 차를 주욱 돌린다.
 대원들은 찻잔을 받아 들고 앉아서 무릎 위에 올려놓은 과자 접시를 들여다보면서,  '에게 ―--- 요걸루 어디 간에 기별이나 가겠나.’
 하는 듯한 표정을 지으며 입맛을 다신다.
 장내는 사기 그릇이 부딪쳐 대그락거리는 소리와 잡담을 하는 소리로 웅성웅성하는데, 맨 앞줄 한구석에서 하와이안 기타를 뜯는 소리가 모기 소리처럼 애응애응 하고 들리기 시작한 다.
 남양의 달밤을 상상케 하는 애련하고도 청아한 선율에, 회장은 다시 조용해졌다. C 전문의 명물인 익살꾼으로 기타의 명수인 S군이 자청을 해서 한 곡조를 타는 것이다.
 S군은 한참 타다가 저 혼자 신이 나서 악기를 들고 일어나 엉덩춤을 춘다. 메기 같은 넓적한 입을 실룩거리며 토인의 노래를 흉내내는데, 그 목소리는 체수에 어울리지 않게, 염생이가 우는 소리와 흡사하게 떨려 나와서, 여러 사람의 웃음보가 터졌다. 어떤 중학생은 웃음을 억지로 참다가, 입에 물고 있던 과자를 앞줄에 앉은 사람의 뒤통수에다가 확 내뿜었다. 한구석에 몰려 앉은 여학생들은 손수건을 입에다 대고 허리를 잡는다.
 "재청요―---"
 "앙코르―--- 앙코르 ―--- "하는 소리가 여기저기서 일어나며 회장 안은 벌통 속처럼 와글와글한다. S군은 저더러 잘 한다는 줄만 알고, 두번 세번 껑충거리고 나와서 익살을 깨트리는 바람에 점잔을 빼던 사회자도 간신히 웃음을 참고 앉았다. 그는 미소를 띠고 일어서며,
 "여러분 고만 조용헙시다."
 하고 손을 들었다.
 "지금부터 여러분의 체험담을 듣겠습니다. 한 사람도 빼어 놓지 않고 고향에서 활동 하던 이야기를 골고루 듣고는 싶지만, 시간이 허락지 않는 관계로 유감천만이나 사회자가 몇 분을 지적할 수밖에 없습니다."
 하고 양복 주머니에서 각 지방으로부터 온 통신과, 이미 신문에 발표된 대원들의 보고서를 한 뭉텅이나 꺼내 놓고 뒤적 거리 더니,
 "금년에 활동한 계몽 대원 중에 뛰어나게 좋은 성적을 보여 주었을 뿐 아니라, 글을 깨쳐 준 아동의 수효로는 우리 신문사에서 이 운동을 개시한 이래 최고 기록을 지은 분을 소개 하겠소이다."
 하고는 다시 안경 너머로 서류를 들여다보다가 얼굴을 들고 선생이 출석부를 부르듯이,
 "×× 고등 농림의 박동혁(朴東赫) 군!"
 하고 목소리를 높였다. 장내는 테를 메인 듯이 긴장해졌건만, 제 이름을 못 들었는지 얼핏 대답하는 사람이 없다.
 "박동혁 군 왔소?"
 사회자는 더한층 목소리를 높이고는 사면을 살핀다. 만장의 학생들은, ' 박동혁이가 어떻게 생긴 사람이야!’
 하는 듯이 서로 돌려다보며 이름을 불린 고농 학생을 찾는다.
 "여기 있습니다."
 맨 뒷줄에서 굵다란 목소리가 청처짐하게 들렸다. 여러 사람의 고개는 일제히 목소리가 난데로 돌려졌다."""
        
    # 문단/문장 단위 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,      # 한 청크의 목표 글자 수
        chunk_overlap=30,    # 앞뒤 내용 연결을 위한 중복 구간
        separators=[
            r"(?<=[.?!])\s+",   # 1순위: 문장 끝 (마침표+공백)\
            "\n",               # 2순위: 줄바꿈
            "\n\n",             # 3순위: 문단 바꿈
        ],
        is_separator_regex=True, #정규표현식 사용 여부
        length_function=len,
    )
        
    chunks = splitter.split_text(text)

    print(f"len(chunks): {len(chunks)}")
    summary, results = run_manual_episode(chunks, episode_seq=1, clear_after=False, conflict_log_path="conflicts.csv")
    print(summary)
    print("main에서는 run_manual_episode([...])를 호출하도록 text 변수 등을 설정하세요.")
