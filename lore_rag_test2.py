import os
import json
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

# ============================
# 0. 경로 / 환경 설정
# ============================
os.environ['OPENAI_API_KEY'] = "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" 

STORY_FILE = "story.txt"
GROUND_TRUTH_FILE = "teststory10.json"
VECTOR_DIR = "./chroma_story_rag"

# ============================
# 1. in-memory "DB"
# ============================
current_story_db: List[Dict[str, Any]] = []   # 정상 설정 저장
conflict_db: List[Dict[str, Any]] = []        # 충돌 정보 저장
_vectordb: Optional[Chroma] = None            # VectorStore 캐시


# ============================
# 2. VectorStore 준비
# ============================
def build_story_vectordb() -> Chroma:
    """story.txt를 청킹하고 Chroma VectorStore로 만든다."""
    global _vectordb

    if _vectordb is not None:
        return _vectordb

    if not os.path.exists(STORY_FILE):
        raise FileNotFoundError(f"story.txt 를 찾을 수 없습니다: {STORY_FILE}")

    with open(STORY_FILE, "r", encoding="utf-8") as f:
        story_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_text(story_text)
    print(f"[VectorStore] story.txt에서 {len(chunks)}개의 청크 생성")

    embeddings = OpenAIEmbeddings()

    if os.path.exists(VECTOR_DIR):
        print(f"[VectorStore] 기존 벡터 스토어 로드: {VECTOR_DIR}")
        _vectordb = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embeddings,
            collection_name="story_chunks",
        )
    else:
        print(f"[VectorStore] 새 Chroma VectorStore 생성 중...")
        _vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=[{"chunk_index": i} for i in range(len(chunks))],
            persist_directory=VECTOR_DIR,
            collection_name="story_chunks",
        )
        print("[VectorStore] story_chunks 컬렉션 생성 완료")

    return _vectordb


# ============================
# 3. Tools 정의 (Agent가 쓸 도구)
# ============================

@tool
def search_story_context(query: str) -> str:
    """
    [스토리 맥락 검색 도구]
    story.txt 기반 VectorStore에서 관련 문맥을 검색합니다.
    - 충돌 판정 시 근거(evidence)를 찾을 때 사용하세요.
    """
    vectordb = build_story_vectordb()
    docs = vectordb.similarity_search(query, k=5)
    if not docs:
        return "[검색 결과 없음]"

    joined = "\n---\n".join(d.page_content for d in docs)
    return "[검색 결과]\n" + joined


@tool
def get_current_db_settings() -> str:
    """
    [Current_DB 조회 도구]
    지금까지 충돌 없이 저장된 설정들을 조회합니다.
    """
    if not current_story_db:
        return "현재 회차에서 확정된 설정이 아직 없습니다."

    lines = []
    for item in current_story_db[-20:]:
        fact = item.get("fact", {})
        text = item.get("text", "")
        if isinstance(fact, dict):
            fact_str = json.dumps(fact, ensure_ascii=False)
        else:
            fact_str = str(fact)
        lines.append(f"- {fact_str} (원문: {text})")
    return "[Current_DB 설정]\n" + "\n".join(lines)


@tool
def extract_facts(sentence: str) -> str:
    """
    [사실 추출 도구]
    한 문장에서 설정 충돌 검사를 위해 비교 가능한 fact들을 JSON 배열로 추출합니다.

    각 fact는 아래 필드를 가진 객체입니다.
    - subject: 인물/대상 이름
    - predicate: 행동/상태/관계 등의 서술
    - obj: 대상이 있을 경우, 그렇지 않으면 빈 문자열
    - time: 시간 정보(없으면 빈 문자열)
    - location: 장소 정보(없으면 빈 문자열)
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_msg = (
        "너는 웹소설 설정 분석 에이전트야.\n"
        "주어진 한 문장에서 '비교 가능한 객관적 사실'만 추출해.\n"
        "각 fact는 JSON 객체로, 필드는 subject, predicate, obj, time, location 이다.\n"
        "모든 fact를 JSON 배열로만 출력해. 다른 말은 쓰지 마."
    )

    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"문장:\n{sentence}"}
    ])
    raw = resp.content.strip()

    # JSON 배열만 잘라내기 (혹시 LLM이 앞뒤로 말 더 붙이면)
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        return raw[start:end]
    except Exception:
        return "[]"


@tool
def save_to_current_db(sentence: str, facts_json: str) -> str:
    """
    [Current_DB 저장 도구]
    충돌이 없다고 판단된 문장의 fact들을 Current_DB에 저장합니다.
    """
    try:
        facts = json.loads(facts_json) if facts_json else []
        for fact in facts:
            current_story_db.append({
                "text": sentence,
                "fact": fact,
            })
        return f"Current_DB에 {len(facts)}개 fact 저장 완료"
    except Exception as e:
        return f"저장 실패: {str(e)}"


@tool
def report_conflict(sentence: str, conflict_type: str, reason: str, evidence: str) -> str:
    """
    [충돌 리포트 도구]
    감지된 설정 충돌을 Conflict_DB에 저장합니다.

    conflict_type:
      - 'Hard' : 생사/시간/공간/세계관 규칙 같은 객관적 사실 모순
      - 'Soft' : 캐릭터 성격, 관계, 감정 흐름, 말투가 기존 설정과 명확히 상충
    reason:
      - 왜 충돌인지 자연어로 구체적으로 설명
    evidence:
      - search_story_context나 get_current_db_settings 결과에서 그대로 가져온 핵심 근거 문장
    """
    conflict_entry = {
        "text": sentence,
        "conflict_type": conflict_type,
        "reason": reason,
        "evidence": evidence,
    }
    conflict_db.append(conflict_entry)
    return f"Conflict_DB에 {conflict_type} 충돌 1건 저장"


# ============================
# 4. Agent (LLM + Tools) 구현
# ============================

TOOLS = [
    search_story_context,
    get_current_db_settings,
    extract_facts,
    save_to_current_db,
    report_conflict,
]

TOOL_NAME_MAP = {t.name: t for t in TOOLS}


def create_agent_llm(model_name: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model_name, temperature=0)
    return llm.bind_tools(TOOLS)


SYSTEM_PROMPT = (
    "너는 웹소설 설정을 관리하는 자율형 에이전트 'Lore Keeper'야.\n\n"
    "입력으로 **한 문장**이 주어지며, 너의 임무는 이 문장이 기존 스토리(story.txt)와\n"
    "논리적으로 충돌하는지 검사하는 것이다.\n\n"
    "사용할 수 있는 도구:\n"
    "1) extract_facts(sentence): 이 문장에서 비교 가능한 fact들을 JSON 배열로 추출\n"
    "2) search_story_context(query): story.txt VectorStore에서 관련 문맥 검색\n"
    "3) get_current_db_settings(): 지금까지 축적된 Current_DB 설정 조회\n"
    "4) save_to_current_db(sentence, facts_json): 충돌 없을 때 fact들을 Current_DB에 저장\n"
    "5) report_conflict(sentence, conflict_type, reason, evidence): 충돌을 Conflict_DB에 기록\n\n"
    "[작업 절차]\n"
    "1. 먼저 extract_facts를 호출해 이 문장에서 fact들을 뽑아라.\n"
    "2. fact를 요약한 질의를 사용해 search_story_context를 반드시 최소 1회 이상 호출해라.\n"
    "3. 필요하다면 get_current_db_settings도 호출해라.\n"
    "4. fact들과 검색된 맥락, Current_DB를 비교해 **정말로 명확한 모순이 있는 경우에만**\n"
    "   report_conflict를 사용해 Hard 또는 Soft로 기록해라.\n"
    "   - Hard: 생사/시간/공간/세계관 규칙이 기존 텍스트와 정면으로 반대일 때만.\n"
    "   - Soft: 캐릭터 성격, 관계, 감정 흐름, 말투가 기존 텍스트와 뚜렷하게 상반될 때만.\n"
    "   - 애매하거나 해석의 여지가 있으면 '충돌 없음'으로 처리해야 한다.\n"
    "5. 명확한 근거가 없으면 report_conflict를 호출하지 말고, save_to_current_db로 fact를 저장해라.\n"
    "6. report_conflict를 호출할 때 evidence에는 search_story_context나 Current_DB 결과에서\n"
    "   가져온 한두 문장을 그대로 넣어라. 입력 문장 자체를 evidence로 쓰지 마라.\n\n"
    "최종적으로 너의 자연어 답변에는 다음을 포함해라.\n"
    "- 이 문장이 충돌인지 여부 (충돌 없음 / Hard Conflict / Soft Conflict)\n"
    "- 그 이유 요약 (1~3문장)\n"
    "- 사용한 evidence 요약 (어떤 근거 문장을 참고했는지)\n"
)


def run_agent_on_sentence(llm_with_tools, sentence: str, max_steps: int = 4) -> Dict[str, Any]:
    """
    ReAct 스타일 도구 호출 루프:
    - LLM이 직접 어떤 tool을 쓸지 결정
    - tool 호출 결과를 ToolMessage로 다시 넘겨서 여러 단계 추론
    """
    messages: List[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"다음 문장이 기존 설정과 충돌하는지 검사해줘:\n{sentence}")
    ]

    final_answer: str = ""
    for step in range(max_steps):
        ai_msg: AIMessage = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        tool_calls = ai_msg.tool_calls or []
        if not tool_calls:
            # 더 이상 tool 호출 없이 답변 끝
            final_answer = ai_msg.content
            break

        # tool 호출들 실행
        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call["args"] or {}
            tool_fn = TOOL_NAME_MAP.get(tool_name)
            if tool_fn is None:
                tool_result = f"[ERROR] Unknown tool: {tool_name}"
            else:
                try:
                    tool_result = tool_fn.invoke(tool_args)
                except Exception as e:
                    tool_result = f"[ERROR] Tool 실행 실패: {str(e)}"
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    name=tool_name,
                    tool_call_id=call["id"],
                )
            )

    # 이 문장에 대해 Conflict_DB에 저장된 것이 있는지 확인
    is_conflict = False
    conflict_type = "None"
    evidence = ""
    reason = ""

    for c in conflict_db[::-1]:  # 뒤에서부터 (가장 최근)
        if c.get("text") == sentence:
            is_conflict = True
            conflict_type = c.get("conflict_type", "None")
            evidence = c.get("evidence", "")
            reason = c.get("reason", "")
            break

    # 타입 정규화
    ct_norm = conflict_type.lower()
    if "hard" in ct_norm:
        conflict_type = "Hard"
    elif "soft" in ct_norm:
        conflict_type = "Soft"
    else:
        conflict_type = "None"

    return {
        "sentence": sentence,
        "is_conflict": is_conflict,
        "conflict_type": conflict_type,
        "evidence": evidence,
        "reason": reason,
        "final_answer": final_answer,
    }


# ============================
# 5. 메트릭 계산
# ============================

def compute_metrics(results: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]):
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

        if y_true == y_pred:
            correct_conflict += 1

        if y_pred and y_true:
            tp += 1
        elif y_pred and not y_true:
            fp += 1
        elif (not y_pred) and y_true:
            fn += 1

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


# ============================
# 6. 전체 실험 실행
# ============================

def run_experiment():
    print("🤖 에이전틱 Lore Keeper + RAG 실험 시작...\n")

    # 1) VectorStore 준비
    print("[1단계] VectorStore 빌드/로딩 중...")
    build_story_vectordb()
    print("✓ VectorStore 준비 완료\n")

    # 2) Agent LLM 생성 (도구 바인딩)
    print("[2단계] Agent LLM 생성 중...")
    llm_with_tools = create_agent_llm()
    print("✓ Agent LLM 준비 완료\n")

    # 3) Ground Truth 로드
    if not os.path.exists(GROUND_TRUTH_FILE):
        raise FileNotFoundError(f"ground_truth 파일을 찾을 수 없습니다: {GROUND_TRUTH_FILE}")

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    print(f"[3단계] Ground Truth 로드 완료: {len(gt_data)}개 샘플\n")

    # 4) 각 샘플 문장에 대해 에이전트 실행
    print("[4단계] 샘플 처리 시작...\n")
    results: List[Dict[str, Any]] = []

    for i, sample in enumerate(gt_data, start=1):
        sentence = sample["input_text"]
        print(f"--- [{i}/{len(gt_data)}] ---")
        print("입력 문장:", sentence, "\n")

        result = run_agent_on_sentence(llm_with_tools, sentence)
        results.append(result)

    # 5) 메트릭 계산
    print("[5단계] 메트릭 계산 중...")
    compute_metrics(results, gt_data)

    # 6) Conflict_DB / Current_DB 요약
    print("[Conflict_DB 요약]")
    print(f"- 감지된 충돌 수: {len(conflict_db)}")
    for i, c in enumerate(conflict_db[:10], start=1):
        print(f"  ({i}) [{c.get('conflict_type', 'Unknown')}] {c.get('reason', '')}")
        print(f"      문장: {c.get('text', '')}")
        print(f"      evidence: {c.get('evidence', '')}\n")

    print("[Current_DB 요약]")
    print(f"- 저장된 설정 수: {len(current_story_db)}")
    for i, item in enumerate(current_story_db[:10], start=1):
        fact = item.get("fact", {})
        print(f"  ({i}) fact: {fact}")
        print(f"      문장: {item.get('text', '')}\n")


if __name__ == "__main__":
    run_experiment()
