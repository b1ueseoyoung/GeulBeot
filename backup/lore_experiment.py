import os
import json
from typing import List, Dict, Any, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# 0. 경로 및 전역 설정
# ==========================================
STORY_PATH = "story.txt"                    # 전체 소설 원문
GROUND_TRUTH_PATH = "ground_truth_100_v3.json"  # 정답지 100개
STORY_DB_PATH = "./chroma_db_story"         # story_chunks 벡터 스토어 경로

# ⚠️ 반드시 셸에서 미리 설정해 두기 (예: export OPENAI_API_KEY=...)
# os.environ["OPENAI_API_KEY"] = "sk-..."  # <- 코드 안에서는 쓰지 말기!


# ==========================================
# 1. 데이터베이스를 흉내내는 in-memory 구조
#    (논문에서는 RDBMS + VectorStore로 설명)
# ==========================================
# Current_DB: 이번 회차에서 확정된 설정들
current_story_db: List[Dict[str, Any]] = []      # {"text", "fact"}

# Conflict_DB: 감지된 충돌들
conflict_report_db: List[Dict[str, Any]] = []    # {"text", "type", "reason"}

# Full_Story_DB: 충돌 없이 통과된 회차(여기선 실험이라 의미만 유지)
full_story_db: List[Dict[str, Any]] = []


# ==========================================
# 2. 소설 전체를 청킹 + 벡터화 (5.1.1)
# ==========================================
def build_story_vectordb(story_path: str, persist_dir: str) -> Tuple[Chroma, int]:
    """전체 소설을 읽어서 story_chunks 컬렉션으로 벡터화.
    - 논문 5.1.1: '원고를 문단 단위로 청킹 및 벡터화하여 Vector Store에 적재'
    """
    with open(story_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100
    )
    chunks = splitter.split_text(full_text)

    print(f"[VectorStore] story_chunks에 {len(chunks)}개 청크 적재 준비 중...")

    embedding = OpenAIEmbeddings()

    # Chroma 0.4.x 이상에서는 persist()가 자동이지만, 경로 지정은 동일
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
        metadatas=[{"chunk_index": i} for i in range(len(chunks))],
        persist_directory=persist_dir,
        collection_name="story_chunks",
    )

    print(f"[VectorStore] story_chunks에 {len(chunks)}개 청크 적재 완료\n")
    return vectordb, len(chunks)


# ==========================================
# 3. LoreKeeperAgent: Agentic AI (MCP 서버 역할 가정)
# ==========================================
class LoreKeeperAgent:
    """
    논문에서 말한 'PlayMCP 기반 LLM 에이전트'의 실제 구현 버전.
    - 이 클래스 하나가 MCP 서버 내부의 '설정 충돌 검사 도구' 역할을 한다고 보면 됨.
    - LangChain의 AgentExecutor는 사용하지 않고, 직접 LLM 호출 로직만 구현.
    """

    def __init__(self, vectordb: Chroma, model_name: str = "gpt-4o-mini"):
        # 분류/충돌분석에 사용할 LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.vectordb = vectordb

    # ---------- 5.1.1: A/B/C/D 분류 ----------
    def classify_chunk(self, chunk: str) -> str:
        """
        chunk를 A/B/C/D 중 하나로 분류.
        - A: 사실/설정
        - B: 감정/내면
        - C: 대화
        - D: 단순 서술
        """
        messages = [
            (
                "system",
                "너는 소설 문장을 설정 유형으로 분류하는 분류기다. "
                "다음 네 가지 중 하나만 출력해라: A, B, C, D.\n"
                "- A: 세계관/규칙/인물·아이템의 상태 등 '사실/설정'\n"
                "- B: 등장인물의 감정·내면 독백\n"
                "- C: 등장인물의 대사(따옴표로 된 말)\n"
                "- D: 배경·풍경·움직임 등 단순 묘사(설정과 직접 관계 없음)\n"
                "출력은 반드시 알파벳 한 글자(A/B/C/D)만 포함해야 한다."
            ),
            (
                "user",
                f"다음 문장의 타입을 A/B/C/D 중 하나로만 답해라.\n\n문장: ```{chunk}```"
            ),
        ]
        res = self.llm.invoke(messages)
        t = res.content.strip().upper()
        # 방어적으로 첫 글자만 사용
        if t and t[0] in ["A", "B", "C", "D"]:
            return t[0]
        return "D"  # 이상하면 그냥 D로 처리

    # ---------- RAG 검색 (Lore_DB + Current_DB를 묶어서 흉내) ----------
    def search_knowledge(self, chunk: str) -> str:
        """
        5.1.2에서 말하는:
        - 벡터 스토어(RAG)에서 관련 설정 검색
        - Current_DB(이미 확정된 설정들)에서 최근 설정 가져오기
        """
        # story_chunks에서 유사 문맥 검색
        docs = self.vectordb.similarity_search(chunk, k=5)
        story_context = "\n\n".join([d.page_content for d in docs])

        # Current_DB 최신 설정들
        if current_story_db:
            current_context = "\n".join(
                [f"- {item['fact']} (원문: {item['text']})"
                 for item in current_story_db[-10:]]
            )
        else:
            current_context = "현재 회차에서 확정된 설정이 아직 없습니다."

        return (
            "[story_chunks에서 검색된 문맥]\n"
            f"{story_context}\n\n"
            "[Current_DB에 누적된 설정]\n"
            f"{current_context}"
        )

    # ---------- 5.1.2: Hard/Soft Conflict 판정 ----------
    def analyze_conflict(self, chunk: str, rag_context: str) -> Dict[str, Any]:
        """
        RAG 결과 + 현재 청크를 보고 Hard/Soft Conflict 여부를 JSON으로 반환.
        - is_conflict: bool
        - conflict_type: "Hard" / "Soft" / "None"
        - reason: 한국어 설명
        - fact_summary: 충돌 없을 때 Current_DB에 저장할 핵심 설정 요약
        """
        system_msg = (
            "너는 웹소설의 설정 충돌을 검사하는 'Lore Keeper' 에이전트다.\n"
            "입력으로 현재 문장(Chunk)과 RAG로 검색된 기존 설정 문맥이 주어진다.\n\n"
            "[판정 기준]\n"
            "1) Hard Conflict:\n"
            "   - 이미 죽은 인물이 다시 살아 등장\n"
            "   - 세계관의 법칙(RULE)을 정면으로 위반\n"
            "   - 시간축 상 절대 일어날 수 없는 사건(동시간에 두 장소에 동시에 존재 등)\n"
            "2) Soft Conflict:\n"
            "   - 캐릭터 성격, 관계, 감정 흐름이 이전 설정과 크게 어긋남\n"
            "   - 말투/톤이 급격하게 바뀌어 독자에게 큰 위화감을 주는 경우\n\n"
            "출력은 반드시 다음 JSON 형식 하나로만 반환해라.\n"
            '{\n'
            '  "is_conflict": true 또는 false,\n'
            '  "conflict_type": "Hard" 또는 "Soft" 또는 "None",\n'
            '  "reason": "한국어로 충돌 여부 및 사유 설명",\n'
            '  "fact_summary": "충돌이 없다면, 이번 문장에서 추출한 핵심 설정 요약(없으면 빈 문자열)"\n'
            '}\n'
        )

        user_msg = (
            "다음은 기존 설정(RAG 검색 결과)과 현재 검사할 문장이다.\n\n"
            f"[RAG 검색 결과]\n{rag_context}\n\n"
            f"[현재 Chunk]\n{chunk}\n\n"
            "위 정보를 바탕으로 설정 충돌 여부를 판정하고, 반드시 JSON만 출력해라."
        )

        res = self.llm.invoke(
            [
                ("system", system_msg),
                ("user", user_msg),
            ]
        )
        raw = res.content.strip()

        # JSON 파싱 (혹시라도 앞뒤에 설명이 붙으면 {} 범위만 잘라내기)
        def _safe_parse_json(text: str) -> Dict[str, Any]:
            try:
                return json.loads(text)
            except Exception:
                # 첫 {, 마지막 } 기준으로 재시도
                if "{" in text and "}" in text:
                    s = text.find("{")
                    e = text.rfind("}")
                    try:
                        return json.loads(text[s:e + 1])
                    except Exception:
                        pass
            # 실패 시 기본값
            return {
                "is_conflict": False,
                "conflict_type": "None",
                "reason": f"JSON 파싱 실패, 원본 응답: {text}",
                "fact_summary": "",
            }

        parsed = _safe_parse_json(raw)

        # 필드 보정
        parsed.setdefault("is_conflict", False)
        parsed.setdefault("conflict_type", "None")
        parsed.setdefault("reason", "")
        parsed.setdefault("fact_summary", "")

        # 타입 정리
        parsed["is_conflict"] = bool(parsed["is_conflict"])
        if parsed["conflict_type"] not in ["Hard", "Soft", "None"]:
            parsed["conflict_type"] = "None"

        return parsed

    # ---------- 전체 파이프라인: 한 문장(chunk)에 대해 5.1.1~5.1.3 수행 ----------
    def analyze_chunk(self, chunk: str) -> Dict[str, Any]:
        """
        하나의 chunk에 대해:
        1) A/B/C/D 분류
        2) D면 바로 PASS (is_conflict=False)
        3) A/B/C면 RAG 검색 + Hard/Soft Conflict 분석
        4) 결과에 따라 Current_DB 또는 Conflict_DB 업데이트
        """
        chunk_type = self.classify_chunk(chunk)

        # D 타입이면 아무 것도 하지 않고 PASS
        if chunk_type == "D":
            return {
                "chunk_type": "D",
                "is_conflict": False,
                "conflict_type": "None",
                "reason": "단순 서술(D)로 판단되어 충돌 검사 생략.",
                "fact_summary": "",
            }

        # A/B/C 타입이면 RAG 검색
        rag_context = self.search_knowledge(chunk)

        # Hard/Soft Conflict 분석
        analysis = self.analyze_conflict(chunk, rag_context)
        is_conflict = analysis["is_conflict"]
        conflict_type = analysis["conflict_type"]
        reason = analysis["reason"]
        fact_summary = analysis["fact_summary"]

        if is_conflict:
            conflict_report_db.append(
                {
                    "text": chunk,
                    "type": conflict_type,
                    "reason": reason,
                }
            )
        else:
            if fact_summary:
                current_story_db.append(
                    {
                        "text": chunk,
                        "fact": fact_summary,
                    }
                )

        return {
            "chunk_type": chunk_type,
            "is_conflict": is_conflict,
            "conflict_type": conflict_type,
            "reason": reason,
            "fact_summary": fact_summary,
        }


# ==========================================
# 4. 실험 루틴
#    - ground_truth_100_v3.json 기준으로 100개 샘플 평가
# ==========================================
def load_ground_truth(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_metrics(results: List[Dict[str, Any]]) -> None:
    """
    results 리스트는 다음 필드를 포함한다고 가정:
    - gt_is_conflict, pred_is_conflict (bool)
    - gt_type, pred_type ("Hard Conflict"/"Soft Conflict"/"None")
    """
    n = len(results)
    correct_is_conflict = sum(
        1 for r in results if r["gt_is_conflict"] == r["pred_is_conflict"]
    )

    # 이진 분류 기준 (conflict 여부)
    tp = sum(
        1
        for r in results
        if r["gt_is_conflict"] and r["pred_is_conflict"]
    )
    fp = sum(
        1
        for r in results
        if (not r["gt_is_conflict"]) and r["pred_is_conflict"]
    )
    fn = sum(
        1
        for r in results
        if r["gt_is_conflict"] and (not r["pred_is_conflict"])
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    acc_is_conflict = correct_is_conflict / n if n > 0 else 0.0

    # Hard/Soft 타입 정확도 (충돌이라고 판정된 것들에 대해서만)
    type_samples = [
        r for r in results if r["gt_is_conflict"]
    ]
    if type_samples:
        correct_type = sum(
            1
            for r in type_samples
            if (
                r["pred_is_conflict"]
                and r["gt_type"].startswith(r["pred_type"])
            )
        )
        type_acc = correct_type / len(type_samples)
    else:
        type_acc = 0.0

    print("\n================ [실험 결과 요약] ================")
    print(f"- 샘플 수: {n}")
    print(f"- Conflict 여부 정확도: {acc_is_conflict:.3f}")
    print(f"- Precision (conflict): {precision:.3f}")
    print(f"- Recall (conflict):    {recall:.3f}")
    print(f"- F1 (conflict):        {f1:.3f}")
    print(f"- Hard/Soft 타입 정확도(충돌인 샘플 기준): {type_acc:.3f}")
    print("=================================================\n")


def run_experiment():
    # 1) story.txt 를 벡터화
    vectordb, chunk_count = build_story_vectordb(STORY_PATH, STORY_DB_PATH)

    # 2) LoreKeeperAgent 생성
    agent = LoreKeeperAgent(vectordb=vectordb, model_name="gpt-4o-mini")

    # 3) 정답지 로드
    gt_data = load_ground_truth(GROUND_TRUTH_PATH)
    print(f"[실험 시작] 샘플 개수: {len(gt_data)}\n")

    results = []

    for idx, item in enumerate(gt_data, start=1):
        text = item["input_text"]
        gt_is_conflict = bool(item["is_conflict"])
        gt_type_str = item.get("conflict_type", "None")

        print(f"--- [{idx}/{len(gt_data)}] ---")
        print(f"입력 문장: {text}")

        result = agent.analyze_chunk(text)

        print(f"  → 분류 결과: {result['chunk_type']}")
        print(f"  → is_conflict: {result['is_conflict']} ({result['conflict_type']})")
        print(f"  → reason: {result['reason']}\n")

        results.append(
            {
                "text": text,
                "gt_is_conflict": gt_is_conflict,
                "gt_type": gt_type_str,
                "pred_is_conflict": result["is_conflict"],
                "pred_type": (
                    "Hard Conflict"
                    if result["conflict_type"] == "Hard"
                    else "Soft Conflict"
                    if result["conflict_type"] == "Soft"
                    else "None"
                ),
            }
        )

    # 4) 메트릭 계산
    compute_metrics(results)

    # 5) Conflict_DB / Current_DB 상태 간단 요약 (논문용 설명에 활용 가능)
    print("[Conflict_DB 요약]")
    print(f"- 감지된 충돌 수: {len(conflict_report_db)}")
    if conflict_report_db:
        for i, c in enumerate(conflict_report_db[:5], start=1):
            print(f"  ({i}) [{c['type']}] {c['reason']}")
            print(f"      문장: {c['text']}\n")

    print("[Current_DB 요약]")
    print(f"- 저장된 설정 수: {len(current_story_db)}")
    if current_story_db:
        for i, s in enumerate(current_story_db[:5], start=1):
            print(f"  ({i}) fact: {s['fact']}")
            print(f"      문장: {s['text']}\n")


# ==========================================
# 5. main
# ==========================================
if __name__ == "__main__":
    run_experiment()
