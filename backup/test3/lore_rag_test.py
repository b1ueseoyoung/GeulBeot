import os
import json
from typing import List, Dict, Any, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========= 경로 설정 =========
os.environ['OPENAI_API_KEY'] = "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" 
STORY_FILE = "story.txt"                     # 네가 가진 원문
GROUND_TRUTH_FILE = "ground_truth_100_v3.json"
VECTOR_DIR = "./chroma_story_rag"           # story용 VectorStore 디렉토리

# ========= in-memory "DB" 구조 =========
current_story_db: List[Dict[str, Any]] = []   # 충돌 없다고 통과된 설정들
conflict_db: List[Dict[str, Any]] = []        # 감지된 충돌들
full_story_db: List[Dict[str, Any]] = []      # 전체 통과 시 저장되는 회차 (여기선 실험이라 의미만)


# ========= 0. VectorStore 준비 (story.txt → chunk → Chroma) =========
def build_story_vectordb() -> Chroma:
    """story.txt를 청킹하고 Chroma VectorStore로 만든다."""
    if not os.path.exists(STORY_FILE):
        raise FileNotFoundError(f"story.txt 를 찾을 수 없습니다: {STORY_FILE}")

    with open(STORY_FILE, "r", encoding="utf-8") as f:
        story_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_text(story_text)

    print(f"[VectorStore] story.txt에서 {len(chunks)}개의 청크 생성")

    embeddings = OpenAIEmbeddings()

    # 새로 생성 (매번 덮어쓰기 싫으면 조건문으로 분기해도 됨)
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=[{"chunk_index": i} for i in range(len(chunks))],
        persist_directory=VECTOR_DIR,
        collection_name="story_chunks",
    )
    print("[VectorStore] story_chunks 컬렉션 생성 완료")
    return vectordb


# ========= 1. 에이전트 클래스 (Agentic Lore Keeper) =========
class LoreKeeperAgent:
    """
    논문에서 말하는 'PlayMCP 기반 에이전트 AI'를 코드로 구현한 형태.
    - LLM 여러 개 호출 (분류, 사실 추출, 충돌 판정)
    - 내부적으로 RAG(VectorStore + Current_DB) 사용
    """

    def __init__(self, vectordb: Chroma, model_name: str = "gpt-4o"):
        self.vectordb = vectordb
        # 분류 / 사실추출 / 충돌판정은 같은 모델을 재사용해도 됨
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    # ----- RAG용 컨텍스트 조회 -----
    def _retrieve_story_context(self, chunk: str, k: int = 5) -> str:
        """현재 chunk를 기준으로 story VectorStore에서 유사한 부분을 가져온다."""
        docs = self.vectordb.similarity_search(chunk, k=k)
        context = "\n\n".join([d.page_content for d in docs])
        return context

    def _current_db_context(self) -> str:
        """Current_DB에 저장된 설정들을 텍스트로 합친다."""
        if not current_story_db:
            return "현재 회차에서 확정된 설정이 아직 없습니다."
        lines = []
        for item in current_story_db[-20:]:
            fact = item.get("fact", "")
            text = item.get("text", "")
            lines.append(f"- {fact} (원문: {text})")
        return "\n".join(lines)

    # ----- 1단계: A/B/C/D 분류 (RAG 포함) -----
    def classify_chunk(self, chunk: str) -> str:
        """
        chunk를 A/B/C/D로 분류.
        이때 story 전체(RAG) + Current_DB까지 참고.
        """
        story_ctx = self._retrieve_story_context(chunk, k=5)
        current_ctx = self._current_db_context()

        system_msg = (
            "너는 웹소설 설정을 관리하는 에이전트 AI 'Lore Keeper'야.\n"
            "아래에 주어지는 문장을 다음 네 가지 중 하나로 분류해.\n"
            "A: 사실/설정 (세계관 규칙, 인물/아이템의 상태, 사건, 배경 설정 등)\n"
            "B: 감정/내면 (인물의 감정, 심리, 내면 독백 등)\n"
            "C: 대화 (직접 화법 대사)\n"
            "D: 단순 서술 (설정과 크게 상관없는 묘사, 분위기 설명 등)\n"
            "반드시 한 글자(A/B/C/D)만 출력해."
        )

        user_msg = (
            "【분류 대상 문장】\n"
            f"{chunk}\n\n"
            "【참고용 이야기 전체 맥락 (RAG 검색 결과)】\n"
            f"{story_ctx}\n\n"
            "【현재까지 확정된 설정(Current_DB)】\n"
            f"{current_ctx}\n\n"
            "위의 정보들을 참고해서 이 문장을 A/B/C/D 중 하나로 분류해.\n"
            "정답은 대문자 한 글자만(A/B/C/D) 적어."
        )

        resp = self.llm.invoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])
        text = resp.content.strip().upper()

        for ch in ["A", "B", "C", "D"]:
            if ch in text:
                return ch
        return "D"  # 안전빵 기본값

    # ----- 2단계: 사실 추출 -----
    def extract_facts(self, chunk: str, chunk_type: str) -> List[Dict[str, Any]]:
        """
        A/B/C 타입에 대해서 비교 가능한 '사실 리스트'를 JSON으로 추출.
        """
        if chunk_type == "D":
            return []

        story_ctx = self._retrieve_story_context(chunk, k=5)
        current_ctx = self._current_db_context()

        system_msg = (
            "너는 웹소설 설정 관리용 분석 에이전트야.\n"
            "주어진 문장과 맥락을 보고, 설정 충돌 검사를 위해 비교 가능한 사실들을 JSON 배열로 추출해.\n"
            "각 사실은 다음 필드를 가진 객체야:\n"
            "- subject: 인물/대상 이름\n"
            "- predicate: 행동/상태/관계 등에 대한 서술\n"
            "- obj: 대상이 있는 경우 그 대상 (없으면 빈 문자열)\n"
            "- category: FACT / EMOTION / DIALOGUE 중 하나\n"
            "- time: 시간 정보가 있으면 자연어로 (없으면 빈 문자열)\n"
            "- location: 장소 정보가 있으면 자연어로 (없으면 빈 문자열)\n"
            "반드시 JSON 배열만 출력해. 다른 말은 쓰지 마."
        )

        user_msg = (
            "【분석 대상 문장】\n"
            f"{chunk}\n\n"
            "【참고용 story RAG 맥락】\n"
            f"{story_ctx}\n\n"
            "【현재까지 확정된 설정(Current_DB)】\n"
            f"{current_ctx}\n\n"
            "위 정보를 참고해서 비교 가능한 사실들을 JSON 배열로 추출해줘."
        )

        resp = self.llm.invoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])
        raw = resp.content.strip()

        # LLM이 코드블럭이나 잡다한 걸 붙일 수 있으니, 대략적으로 [ ... ] 부분만 파싱
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            json_str = raw[start:end]
            facts = json.loads(json_str)
            if isinstance(facts, list):
                return facts
            return []
        except Exception:
            # 파싱 실패하면 그냥 빈 리스트
            return []

    # ----- 3단계: RAG 기반 설정 검색 -----
    def retrieve_related_settings(self, facts: List[Dict[str, Any]], top_k: int = 5) -> str:
        """
        추출된 fact들을 query로 삼아서 story VectorStore + Current_DB에서 관련 내용을 찾는다.
        """
        story_snippets = []
        for fact in facts:
            q = " ".join([
                str(fact.get("subject", "")),
                str(fact.get("predicate", "")),
                str(fact.get("obj", "")),
            ]).strip()
            if not q:
                continue
            docs = self.vectordb.similarity_search(q, k=top_k)
            for d in docs:
                story_snippets.append(d.page_content)

        story_part = "\n\n".join(story_snippets[:10]) if story_snippets else "관련 story 조각 없음."
        current_ctx = self._current_db_context()

        return (
            "【Story VectorStore에서 검색된 관련 문맥】\n"
            f"{story_part}\n\n"
            "【Current_DB에 누적된 설정】\n"
            f"{current_ctx}"
        )

    # ----- 4단계: Hard / Soft Conflict 판정 -----
    def detect_conflict(
        self,
        chunk: str,
        facts: List[Dict[str, Any]],
        rag_context: str
    ) -> Dict[str, Any]:
        """
        Hard/Soft Conflict 여부와 사유를 JSON으로 반환.
        """
        system_msg = (
            "너는 웹소설 설정 충돌을 검사하는 에이전트야.\n"
            "입력 문장과 추출된 사실들, 그리고 기존 설정(RAG 컨텍스트)를 비교하여\n"
            "Hard Conflict / Soft Conflict / None 중에서 판정해.\n\n"
            "- Hard Conflict 예시:\n"
            "  · 이미 죽은 인물이 다시 살아서 등장\n"
            "  · 세계관의 핵심 규칙(예: '마법을 못 쓴다')을 정면으로 위반\n"
            "  · 시간적으로 절대 불가능한 이동, 사건 순서\n\n"
            "- Soft Conflict 예시:\n"
            "  · 캐릭터의 성격, 말투, 관계가 이전과 너무 다르게 묘사되어 위화감\n"
            "  · 감정 흐름, 관계 변화가 뜬금없이 튀어나와서 설득력이 떨어짐\n\n"
            "반드시 다음 키를 가진 JSON 객체만 출력해.\n"
            "is_conflict: true/false\n"
            "conflict_type: \"Hard\" / \"Soft\" / \"None\" 중 하나\n"
            "reason: 한국어 텍스트로 충돌 여부와 이유 설명"
        )

        user_msg = (
            "【입력 문장】\n"
            f"{chunk}\n\n"
            "【추출된 사실(JSON 배열)】\n"
            f"{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
            "【기존 설정 및 RAG 맥락】\n"
            f"{rag_context}\n\n"
            "위 정보를 모두 고려해서 설정 충돌 여부를 판정해줘."
        )

        resp = self.llm.invoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])
        raw = resp.content.strip()

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            json_str = raw[start:end]
            data = json.loads(json_str)
        except Exception:
            # 파싱 실패 시 기본값
            data = {
                "is_conflict": False,
                "conflict_type": "None",
                "reason": "파싱 실패로 인해 충돌 없음으로 처리함.",
            }

        # 보정
        is_conf = bool(data.get("is_conflict", False))
        ctype = data.get("conflict_type", "None")
        if ctype not in ["Hard", "Soft", "None"]:
            ctype = "None"

        data["is_conflict"] = is_conf
        data["conflict_type"] = ctype
        return data

    # ----- chunk 하나 전체 처리 -----
    def process_chunk(self, chunk: str, index: int) -> Dict[str, Any]:
        """
        5.1.1 ~ 5.1.3 한 사이클:
        - 분류
        - (A/B/C이면) 사실 추출 + RAG 검색 + 충돌 판정
        - Conflict_DB / Current_DB에 반영
        """
        print(f"\n--- [Chunk {index+1}] ---")
        print("입력 문장:", chunk)

        chunk_type = self.classify_chunk(chunk)
        print(f"  → 분류 결과: {chunk_type}")

        if chunk_type == "D":
            print("  → 단순 서술(D)로 판단, PASS")
            return {
                "chunk_type": "D",
                "is_conflict": False,
                "conflict_type": "None",
            }

        facts = self.extract_facts(chunk, chunk_type)
        print(f"  → 추출된 사실 개수: {len(facts)}")

        rag_ctx = self.retrieve_related_settings(facts)
        result = self.detect_conflict(chunk, facts, rag_ctx)

        if result["is_conflict"]:
            print(f"  → 설정 충돌 감지! ({result['conflict_type']})")
            conflict_db.append({
                "chunk_index": index,
                "chunk_type": chunk_type,
                "text": chunk,
                "facts": facts,
                "conflict_type": result["conflict_type"],
                "reason": result["reason"],
            })
        else:
            print("  → 충돌 없음. Current_DB에 설정 추가.")
            for fact in facts:
                current_story_db.append({
                    "chunk_index": index,
                    "text": chunk,
                    "fact": fact,
                })

        return {
            "chunk_type": chunk_type,
            "is_conflict": result["is_conflict"],
            "conflict_type": result["conflict_type"],
        }


# ========= 2. 평가(메트릭) 계산 =========
def compute_metrics(results: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]):
    """
    ground_truth_100_v3.json과 비교해서:
    - Conflict 여부 정확도
    - Precision / Recall / F1
    - Hard/Soft 타입 정확도
    """
    assert len(results) == len(ground_truth)

    def norm_conflict_type(s: str) -> str:
        s = (s or "").lower()
        if "hard" in s:
            return "Hard"
        if "soft" in s:
            return "Soft"
        return "None"

    N = len(results)
    correct_conflict = 0
    tp = fp = fn = 0
    correct_type = 0
    true_conflict_count = 0

    for pred, gt in zip(results, ground_truth):
        y_true = bool(gt.get("is_conflict", False))
        y_pred = bool(pred.get("is_conflict", False))

        # conflict 여부 정확도
        if y_true == y_pred:
            correct_conflict += 1

        # Precision/Recall용 TP/FP/FN
        if y_pred and y_true:
            tp += 1
        elif y_pred and not y_true:
            fp += 1
        elif (not y_pred) and y_true:
            fn += 1

        # Hard/Soft 타입 정확도 (진짜 conflict인 경우만)
        if y_true:
            true_conflict_count += 1
            gt_type = norm_conflict_type(gt.get("conflict_type", "None"))
            pred_type = norm_conflict_type(pred.get("conflict_type", "None"))
            if gt_type in ["Hard", "Soft"]:
                if pred_type == gt_type:
                    correct_type += 1

    acc_conflict = correct_conflict / N if N > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    type_acc = correct_type / true_conflict_count if true_conflict_count > 0 else 0.0

    print("\n================ [실험 결과 요약] ================")
    print(f"- 샘플 수: {N}")
    print(f"- Conflict 여부 정확도: {acc_conflict:.3f}")
    print(f"- Precision (conflict): {precision:.3f}")
    print(f"- Recall (conflict):    {recall:.3f}")
    print(f"- F1 (conflict):        {f1:.3f}")
    print(f"- Hard/Soft 타입 정확도(충돌인 샘플 기준): {type_acc:.3f}")
    print("=================================================\n")


# ========= 3. 전체 실험 실행 =========
def run_experiment():
    print("🤖 Agentic Lore Keeper RAG 실험 시작...\n")

    # 1) story VectorStore 구성
    vectordb = build_story_vectordb()

    # 2) ground truth 로드
    if not os.path.exists(GROUND_TRUTH_FILE):
        raise FileNotFoundError(f"ground_truth 파일을 찾을 수 없습니다: {GROUND_TRUTH_FILE}")

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    print(f"[실험 시작] 샘플 개수: {len(gt_data)}\n")

    agent = LoreKeeperAgent(vectordb)
    results: List[Dict[str, Any]] = []

    for i, sample in enumerate(gt_data):
        text = sample["input_text"]
        result = agent.process_chunk(text, index=i)
        results.append(result)

    # 3) 메트릭 계산
    compute_metrics(results, gt_data)

    # 4) Conflict_DB / Current_DB 간단 요약 출력
    print("[Conflict_DB 요약]")
    print(f"- 감지된 충돌 수: {len(conflict_db)}")
    for i, c in enumerate(conflict_db[:10], start=1):
        print(f"  ({i}) [{c['conflict_type']}] {c['reason']}")
        print(f"      문장: {c['text']}\n")

    print("[Current_DB 요약]")
    print(f"- 저장된 설정 수: {len(current_story_db)}")
    for i, item in enumerate(current_story_db[:10], start=1):
        fact = item["fact"]
        print(f"  ({i}) fact: {fact}")
        print(f"      문장: {item['text']}\n")


# ========= 4. main =========
if __name__ == "__main__":
    run_experiment()
