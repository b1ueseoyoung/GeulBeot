import os
import json
from typing import List

# 1. LLM 및 Vector DB 관련
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool

# ==========================================
# 0. 환경 설정
# ==========================================

# ★ 실제 키는 환경변수로 설정하고, 코드에는 넣지 마!
#   터미널에서:
os.environ.setdefault("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Vector Store 경로
LORE_DB_PATH = "./chroma_db_lore"      # lore_items (세계관 설정)
STORY_DB_PATH = "./chroma_db_story"    # story_chunks (원고 문맥)

# 공용 LLM 인스턴스 (필요하면 전역으로 하나만 써도 됨)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ==========================================
# 1. "DB"를 흉내내는 전역 in-memory 구조
#    - 실제 서비스에서는 MySQL + ChromaDB로 분리
# ==========================================
# Script DB / episodes (원고 저장)
episodes_db = []          # {"episode_id", "novel_id", "sequence", "content"}

# 분석 중인 회차에 대한 임시 Current DB
current_story_db = []     # {"text", "fact"}  ← item_type/subject 등 JSON으로 확장 가능

# Conflict DB (논문에서 말하는 Conflict_DB)
conflict_report_db = []   # {"chunk_index", "text", "type", "reason"}

# 전체가 통과됐을 때 누적 저장되는 Full_Story_DB
full_story_db = []        # {"novel_id", "sequence", "content"}


# ==========================================
# 2. 유틸 함수: Script DB 저장 + 청킹 + 벡터화
# ==========================================
def save_script_to_db(manuscript: str, novel_id: int, episode_seq: int) -> int:
    """
    5.1.1 'Script DB에 원고 저장' 단계.
    실제 MySQL episodes 테이블을 in-memory로 흉내냄.
    """
    episode_id = len(episodes_db) + 1
    episodes_db.append({
        "episode_id": episode_id,
        "novel_id": novel_id,
        "sequence": episode_seq,
        "content": manuscript,
    })
    return episode_id


def chunk_and_vectorize_story(manuscript: str, novel_id: int, episode_seq: int) -> List[str]:
    """
    5.1.1: 원고를 문단 단위로 청킹하고, story_chunks 컬렉션에 벡터화하여 저장.
    실제로는 ChromaDB의 story_chunks 컬렉션에 적재해 Co-Author/RAG에 사용.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(manuscript)

    embedding = OpenAIEmbeddings()

    metadatas = [
        {
            "novel_id": novel_id,
            "episode_seq": episode_seq,
            "chunk_index": i,
            "chunk_type": "UNCLASSIFIED",
        }
        for i in range(len(chunks))
    ]

    # story_chunks 컬렉션 생성/갱신
    if os.path.exists(STORY_DB_PATH):
        story_db = Chroma(
            persist_directory=STORY_DB_PATH,
            embedding_function=embedding,
            collection_name="story_chunks",
        )
        story_db.add_texts(texts=chunks, metadatas=metadatas)
    else:
        story_db = Chroma.from_texts(
            texts=chunks,
            embedding=embedding,
            metadatas=metadatas,
            persist_directory=STORY_DB_PATH,
            collection_name="story_chunks",
        )
    story_db.persist()
    return chunks


# ==========================================
# 3. Tools 정의 (MCP 서버 내부 도구 느낌)
# ==========================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    [설정 검색 도구]
    Hard/Soft Conflict를 판단하기 위해
    - Lore_DB(lore_items 컬렉션)과
    - Current_DB(현재 회차에서 확정된 설정들)
    을 함께 조회한다.
    """
    embedding = OpenAIEmbeddings()

    # 1) Lore_DB(lore_items)에서 검색
    if os.path.exists(LORE_DB_PATH):
        lore_db = Chroma(
            persist_directory=LORE_DB_PATH,
            embedding_function=embedding,
            collection_name="lore_items",
        )
        docs = lore_db.similarity_search(query, k=5)
        lore_context = "\n".join([d.page_content for d in docs])
    else:
        lore_context = "기존 Lore_DB(세계관 설정)가 아직 비어 있습니다."

    # 2) Current_DB(이번 회차에서 확정된 설정들) 최근 N개
    if current_story_db:
        recent_current = "\n".join(
            [f"- {item['fact']} (원문: {item['text']})" for item in current_story_db[-10:]]
        )
    else:
        recent_current = "현재 회차에서 확정된 설정이 아직 없습니다."

    return (
        "[Lore_DB에서 검색된 설정]\n"
        f"{lore_context}\n\n"
        "[Current_DB에 누적된 설정]\n"
        f"{recent_current}"
    )


@tool
def report_conflict(chunk_text: str, reason: str, conflict_type: str):
    """
    [충돌 리포트 도구]
    설정 충돌이 감지되었을 때 Conflict_DB에 기록.
    - conflict_type: 'Hard' 또는 'Soft'
    """
    record = {
        "chunk_index": len(conflict_report_db),
        "text": chunk_text,
        "type": conflict_type,
        "reason": reason,
    }
    conflict_report_db.append(record)
    return "충돌 내용이 Conflict DB에 저장되었습니다."


@tool
def save_verified_setting(chunk_text: str, fact_summary: str):
    """
    [정상 설정 저장 도구]
    충돌이 없는 설정/사실/관계(A/B/C 타입)의 핵심 정보를 Current_DB에 저장한다.
    """
    record = {
        "text": chunk_text,
        "fact": fact_summary,  # 실제 구현에선 JSON 파싱 후 DB에 매핑
    }
    current_story_db.append(record)
    return "정상 설정이 Current DB에 저장되었습니다."


# ==========================================
# 4. LLM 호출 유틸: Chunk 타입 분류 + 충돌 분석
#    (여기가 사실상 '에이전트 브레인' 역할)
# ==========================================

def classify_chunk_type(chunk: str) -> str:
    """
    5.1.1: A/B/C/D 분류.
    LLM에게 딱 한 글자(A/B/C/D)만 리턴하게 함.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "너는 웹소설 원고를 분석해서 문장을 A/B/C/D로 분류하는 분류기야.\n"
                "아무 설명 없이, 반드시 'A', 'B', 'C', 'D' 중 한 글자만 출력해.\n\n"
                "A: 사실/설정 (세계관 규칙, 인물/아이템 상태, 능력 등)\n"
                "B: 감정/내면 (감정 묘사, 심리, 관계 변화 등)\n"
                "C: 대화 (캐릭터의 대사)\n"
                "D: 단순 서술 (배경/풍경, 움직임 등 설정과 직접 연결되지 않는 일반 서술)"
            ),
        },
        {
            "role": "user",
            "content": f"다음 문장을 A/B/C/D 중 하나로 분류해줘.\n\n문장:\n{chunk}",
        },
    ]

    resp = llm.invoke(messages)
    label = resp.content.strip().upper()
    if label and label[0] in ["A", "B", "C", "D"]:
        return label[0]
    # 이상한 답이 오면 그냥 D로 취급
    return "D"


def analyze_conflict(chunk: str, search_context: str) -> dict:
    """
    5.1.2: Hard/Soft Conflict 판정 + 요약.
    LLM에게 JSON만 출력하도록 시킨 뒤 파싱.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "너는 웹소설 설정을 관리하는 자율형 AI 에이전트 'Lore Keeper'야.\n"
                "아래 [입력 Chunk]와 [관련 설정 컨텍스트]를 보고 설정 충돌을 분석해.\n\n"
                "Hard Conflict 예시:\n"
                "- 이미 사망한 인물이 다시 생존 상태로 등장\n"
                "- 세계관의 핵심 규칙(RULE)을 정면으로 위반\n"
                "- 시간축 상으로 절대 일어날 수 없는 사건\n\n"
                "Soft Conflict 예시:\n"
                "- 이전 설정과 다르게 캐릭터 성격/관계/감정 흐름이 심하게 어긋남\n"
                "- 말투가 너무 달라져서 독자에게 위화감을 줄 수준\n\n"
                "다음 형식의 JSON만 출력해:\n"
                '{\n'
                '  \"has_conflict\": true/false,\n'
                '  \"conflict_type\": \"Hard\" 또는 \"Soft\" 또는 \"None\",\n'
                '  \"reason\": \"왜 그렇게 판단했는지 한국어 설명\",\n'
                '  \"fact_summary\": \"충돌이 없다면 현재 Chunk에서 추출한 핵심 설정 요약\"\n'
                "}\n"
                "설명이 길어도 괜찮지만, 반드시 유효한 JSON만 출력해."
            ),
        },
        {
            "role": "user",
            "content": (
                "[입력 Chunk]\n"
                f"{chunk}\n\n"
                "[관련 설정 컨텍스트]\n"
                f"{search_context}"
            ),
        },
    ]

    resp = llm.invoke(messages)
    text = resp.content.strip()

    # JSON 파싱 시도
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 혹시 앞뒤에 쓰잘데기 문장이 붙으면 중괄호 부분만 추출
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                # 더 이상 못 살리면 기본값
                data = {
                    "has_conflict": False,
                    "conflict_type": "None",
                    "reason": "JSON 파싱 실패로 기본값 사용",
                    "fact_summary": "",
                }
        else:
            data = {
                "has_conflict": False,
                "conflict_type": "None",
                "reason": "JSON 파싱 실패로 기본값 사용",
                "fact_summary": "",
            }

    # 안전한 기본값 채우기
    data.setdefault("has_conflict", False)
    data.setdefault("conflict_type", "None")
    data.setdefault("reason", "")
    data.setdefault("fact_summary", "")

    return data


# ==========================================
# 5. 전체 파이프라인 실행 함수
#    - '검사하기' 버튼 클릭 후 비동기 작업이라고 논문에서 설명 가능
# ==========================================
def run_lore_keeper_pipeline(manuscript: str, novel_id: int, episode_seq: int):
    print("🤖 Agentic Lore Keeper 파이프라인 시작...\n")

    # 5.1.1 Script DB 저장
    episode_id = save_script_to_db(manuscript, novel_id, episode_seq)
    print(f"[Script DB] episode_id={episode_id} 로 저장 완료")

    # 5.1.1 청킹 + 벡터화 (story_chunks)
    chunks = chunk_and_vectorize_story(manuscript, novel_id, episode_seq)
    print(f"[청킹] 총 {len(chunks)}개 Chunk 생성 및 story_chunks Vector Store 적재")

    # 5.1.1 ~ 5.1.2 반복: Chunk 단위 LLM 에이전트 호출
    for i, chunk in enumerate(chunks):
        print(f"\n--- [Chunk {i+1}/{len(chunks)} 분석 중] ---")
        print(f"내용: {chunk.strip()}\n")

        # 1) A/B/C/D 분류
        chunk_type = classify_chunk_type(chunk)
        print(f"  → 분류 결과: {chunk_type}")

        if chunk_type == "D":
            print("  → 단순 서술(D)로 판단, PASS\n")
            continue

        # 2) 설정 검색 (Lore_DB + Current_DB)
        search_ctx = search_knowledge_base.invoke({"query": chunk})
        # search_knowledge_base는 @tool 이지만, 여기서는 .invoke 로 직접 호출

        # 3) 충돌 분석 (Hard/Soft 판단 + 요약)
        analysis = analyze_conflict(chunk, search_ctx)
        has_conflict = bool(analysis.get("has_conflict"))
        conflict_type = analysis.get("conflict_type", "None")
        reason = analysis.get("reason", "")
        fact_summary = analysis.get("fact_summary", "")

        if has_conflict and conflict_type in ["Hard", "Soft"]:
            # 충돌 리포트 도구 호출
            report_conflict.invoke({
                "chunk_text": chunk,
                "reason": reason,
                "conflict_type": conflict_type,
            })
            print(f"  → ⚠️ {conflict_type} Conflict 감지: {reason}\n")
        else:
            # 정상 설정 저장 도구 호출
            if fact_summary.strip():
                save_verified_setting.invoke({
                    "chunk_text": chunk,
                    "fact_summary": fact_summary,
                })
                print(f"  → ✅ 충돌 없음, 설정 요약 저장: {fact_summary}\n")
            else:
                print("  → ✅ 충돌 없음, 저장할 설정 요약은 없음\n")

    # 5.1.3 후처리: Conflict 여부에 따라 분기
    print("\n" + "=" * 60)
    print("📊 [최종 결과 리포트]")

    if conflict_report_db:
        print(f"🚨 설정 충돌 발견: {len(conflict_report_db)}건")
        for c in conflict_report_db:
            print(f"- [{c['type']}] {c['reason']}")
            print(f"  · 문제 문장: {c['text']}\n")
        print("→ Conflict_DB에 저장된 내용을 바탕으로 인물/아이템/시간축 기준 종합 리포트를 생성할 수 있습니다.")
    else:
        print("✅ 설정 충돌 없음. 현재 회차의 설정을 Full_Story_DB 및 Lore_DB로 반영합니다.")
        full_story_db.append({
            "novel_id": novel_id,
            "sequence": episode_seq,
            "content": manuscript,
        })
        print(f"Full_Story_DB 누적 회차 수: {len(full_story_db)}")

    print("=" * 60)


# ==========================================
# 6. 로컬 테스트용 main
# ==========================================
def main():
    # 테스트용 가짜 원고
    manuscript = """
    레이븐은 숲속을 걸으며 휘파람을 불었다. (단순 서술)
    "난 절대 마법을 쓰지 않아. 그건 내 신조야." 레이븐이 말했다. (설정: 마법 혐오)
    잠시 후, 레이븐은 손에서 거대한 화염구를 발사했다. (설정 충돌: 마법 사용)
    """

    run_lore_keeper_pipeline(
        manuscript=manuscript,
        novel_id=101,
        episode_seq=1,
    )


if __name__ == "__main__":
    main()
