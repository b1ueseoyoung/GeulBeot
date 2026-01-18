# lore_rag_agent_final.py

import os
import json
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# ====== 환경 설정 ======
# 👉 OpenAI 키는 환경변수로 세팅해두었다고 가정 (export OPENAI_API_KEY=...).
# 필요하면 아래 주석 풀고 직접 넣어도 됨.
# os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"

os.environ['OPENAI_API_KEY'] = "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" 
STORY_CHROMA_DIR = "./chroma_story_chunks"
STORY_COLLECTION_NAME = "story_chunks"
GROUND_TRUTH_FILE = "ground_truth_100_v3.json"

# rate limit 때문에 테스트 샘플 수를 제한 (원하면 100으로 늘리면 됨)
MAX_SAMPLES = 20

# 전역 VectorStore / DB
vectordb: Chroma | None = None
conflict_db: List[Dict[str, Any]] = []


# ====== 1. VectorStore 로딩 ======
def load_story_vectordb() -> Chroma:
    global vectordb
    if vectordb is not None:
        return vectordb

    if not os.path.exists(STORY_CHROMA_DIR):
        raise FileNotFoundError(
            f"Chroma 디렉토리 {STORY_CHROMA_DIR} 가 없습니다. "
            "먼저 story.txt로 벡터스토어를 만드는 스크립트를 실행하세요."
        )

    print("[VectorStore] 기존 Chroma 로딩 중...")
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=STORY_CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=STORY_COLLECTION_NAME,
    )
    print("[VectorStore] 로딩 완료")
    return vectordb


# ====== 2. 도구 정의 ======

@tool
def search_story_context(query: str) -> str:
    """
    [스토리 맥락 검색]
    story.txt로부터 만든 Chroma VectorStore에서 query와 관련된 문맥을 검색합니다.
    - 보통 query에는 '검사할 문장 자체' 또는 그 요약을 넣으세요.
    - 최소 1회는 이 도구를 호출한 뒤에 최종 판단을 내려야 합니다.
    """
    db = load_story_vectordb()
    docs = db.similarity_search(query, k=4)
    if not docs:
        return "관련 맥락을 찾지 못했습니다."
    chunks = []
    for i, d in enumerate(docs):
        meta = d.metadata or {}
        idx = meta.get("chunk_index", i)
        chunks.append(f"[chunk {idx}]\n{d.page_content}")
    return "\n\n".join(chunks)


@tool
def report_conflict_decision(
    is_conflict: bool,
    conflict_type: str,
    reason: str,
    evidence: str
) -> str:
    """
    [최종 판단 도구]
    한 문장에 대한 설정 충돌 여부를 최종 결정할 때 호출합니다.

    Args:
        is_conflict: True = 충돌 있음, False = 충돌 없음
        conflict_type: "Hard", "Soft", 또는 "None"
        reason: 왜 그렇게 판단했는지 한국어로 요약
        evidence: story.txt에서 가져온, 이 판단을 뒷받침하는 핵심 문장(들)

    이 도구는 각 문장에 대해 *최대 1번*만 호출해야 합니다.
    """
    # 이 함수 내부에서는 실제로 DB에 저장만 하고, 판단 결과는 외부 루프에서 읽습니다.
    global conflict_db
    record = {
        "is_conflict": bool(is_conflict),
        "conflict_type": conflict_type,
        "reason": reason,
        "evidence": evidence,
    }
    # 충돌인 경우만 conflict_db에 저장 (리포트용)
    if record["is_conflict"]:
        conflict_db.append(record)
        return f"[저장됨] {conflict_type} 충돌로 기록되었습니다."
    else:
        return "[저장됨] 충돌 없음으로 기록되었습니다."


# ====== 3. Agent LLM + tool-calling 루프 ======

SYSTEM_PROMPT = """
너는 웹소설 설정을 관리하는 에이전틱 AI 'Lore Keeper'야.

너의 임무는 **한 문장**을 보고, story.txt 전체를 기반으로
그 문장이 기존 설정과 **논리적으로 충돌하는지** 판단하는 것이다.

[사용 가능한 도구]
1) search_story_context(query):
   - query와 비슷한 문장을 story.txt 벡터스토어에서 검색해 준다.
   - 보통 query에는 '검사할 문장 자체'를 그대로 넣어라.
2) report_conflict_decision(is_conflict, conflict_type, reason, evidence):
   - 최종 판단을 정리해 기록하는 도구다.
   - 이 도구는 각 문장 당 **한 번만** 호출해야 한다.

[작업 절차]
1. 먼저 search_story_context를 **최소 1회 이상** 호출하여,
   검사할 문장과 관련된 맥락을 story.txt에서 찾아라.
2. 검색된 맥락을 자세히 읽고, 다음 기준으로 판단하라.

   - Hard Conflict:
     · 이미 죽은 인물이 다시 살아 등장
     · 시간/공간상 절대 동시에 있을 수 없는 사건
     · 세계관의 중요한 규칙을 정면으로 위반
   - Soft Conflict:
     · 동일 인물의 성격, 감정 흐름, 관계가 이전 설정과 크게 어긋남
     · 말투/태도가 기존 장면과 전혀 어울리지 않아 독자에게 강한 위화감을 줄 정도

3. **아래 조건을 모두 만족할 때만 충돌로 판단하라.**
   - 검색된 맥락 안에, 검사할 문장과 **같은 인물/대상**이 명시되어 있어야 한다.
   - 그 인물에 대해 **명확히 반대되는 사실**이나 **설정**이 문장 형태로 존재해야 한다.
   - 그 문장을 evidence로 그대로 복사해서 사용해야 한다.

4. 위 조건을 만족하지 못하면, 설령 다소 어색해 보여도
   → "충돌 없음"으로 판단해야 한다.
   (새로운 사건이 추가되거나, 이전에 등장하지 않았던 장면들은
    기본적으로 '충돌 없음'으로 간주하라.)

5. 판단을 마쳤으면 report_conflict_decision 도구를 반드시 호출해라.
   - is_conflict: True/False
   - conflict_type:
       · Hard Conflict인 경우 "Hard"
       · Soft Conflict인 경우 "Soft"
       · 충돌이 없으면 "None"
   - reason: 왜 그렇게 판단했는지 간단히 한국어 서술
   - evidence:
       · search_story_context 결과 중, 판단을 뒷받침하는 핵심 문장(들)을 그대로 넣기
       · 충돌이 없으면, "충돌로 볼 만한 근거가 없었다."처럼 요약

[중요]
- 도구를 쓰지 않고 네가 마음대로 결론을 말하지 마라.
- 반드시 search_story_context → report_conflict_decision 순서로 도구를 사용해라.
- 특히 Hard/Soft로 판단하는 경우, evidence에 반드시 명확한 모순 문장을 포함시켜라.
"""

def create_agent_llm(model_name: str = "gpt-4o") -> ChatOpenAI:
    tools = [search_story_context, report_conflict_decision]
    base_llm = ChatOpenAI(model=model_name, temperature=0)
    llm_with_tools = base_llm.bind_tools(tools)
    return llm_with_tools


def run_agent_on_sentence(llm_with_tools: ChatOpenAI, sentence: str) -> Dict[str, Any]:
    """
    한 문장에 대해:
    - LLM + tools 루프를 돌리고
    - report_conflict_decision 도구 호출 인자를 파싱해서 결과 반환
    """
    messages: List[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "다음 한 문장의 설정 충돌 여부를 위 규칙에 따라 판단해줘.\n\n"
                f"[검사할 문장]\n{sentence.strip()}"
            )
        ),
    ]

    final_decision: Dict[str, Any] | None = None

    # 너무 길어지지 않도록 최대 스텝 제한
    for _ in range(6):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # tool 호출이 없으면 루프 종료
        tool_calls = getattr(ai_msg, "tool_calls", []) or []
        if not tool_calls:
            break

        for tc in tool_calls:
            name = tc["name"]
            args = tc["args"]

            if name == "search_story_context":
                query = args.get("query") or sentence
                context = search_story_context.invoke({"query": query})
                messages.append(
                    ToolMessage(
                        content=context,
                        tool_call_id=tc["id"],
                    )
                )

            elif name == "report_conflict_decision":
                # 최종 판단 저장
                is_conflict = bool(args.get("is_conflict", False))
                conflict_type = args.get("conflict_type", "None")
                reason = args.get("reason", "")
                evidence = args.get("evidence", "")

                # 실제 도구 함수도 한 번 호출해서 conflict_db에 저장
                _ = report_conflict_decision.invoke(args)

                final_decision = {
                    "is_conflict": is_conflict,
                    "conflict_type": conflict_type,
                    "reason": reason,
                    "evidence": evidence,
                }

                messages.append(
                    ToolMessage(
                        content="[판단 결과를 기록했습니다.]",
                        tool_call_id=tc["id"],
                    )
                )
                break  # 이 문장에 대한 판단 끝

        if final_decision is not None:
            break

    # 혹시 도구를 전혀 안 쓴 경우 → 기본값: 충돌 없음
    if final_decision is None:
        final_decision = {
            "is_conflict": False,
            "conflict_type": "None",
            "reason": "도구 호출 없이 종료되어, 기본적으로 충돌 없음으로 처리했습니다.",
            "evidence": "",
        }

    return final_decision


# ====== 4. 메트릭 계산 ======

def compute_metrics(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
) -> None:
    assert len(preds) == len(gts)

    def norm_type(s: str) -> str:
        s = (s or "").lower()
        if "hard" in s:
            return "Hard"
        if "soft" in s:
            return "Soft"
        return "None"

    N = len(preds)
    correct_conflict = 0
    tp = fp = fn = 0
    correct_type = 0
    true_conflict_count = 0

    for pred, gt in zip(preds, gts):
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
            gt_type = norm_type(gt.get("conflict_type", "None"))
            pred_type = norm_type(pred.get("conflict_type", "None"))
            if gt_type in ["Hard", "Soft"]:
                if pred_type == gt_type:
                    correct_type += 1

    acc_conflict = correct_conflict / N if N else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    type_acc = correct_type / true_conflict_count if true_conflict_count else 0.0

    print("\n================ [실험 결과 요약] ================")
    print(f"- 샘플 수: {N}")
    print(f"- Conflict 여부 정확도: {acc_conflict:.3f}")
    print(f"- Precision (conflict): {precision:.3f}")
    print(f"- Recall (conflict):    {recall:.3f}")
    print(f"- F1 (conflict):        {f1:.3f}")
    print(f"- Hard/Soft 타입 정확도(충돌인 샘플 기준): {type_acc:.3f}")
    print("=================================================\n")


# ====== 5. 전체 실행 ======

def run_experiment():
    global conflict_db
    conflict_db = []

    print("🤖 에이전틱 Lore Keeper + RAG 최종 실험 시작...\n")

    # 1) 벡터스토어 로딩
    print("[1단계] VectorStore 로딩...")
    load_story_vectordb()
    print("✓ VectorStore 준비 완료\n")

    # 2) Agent LLM 준비
    print("[2단계] Agent LLM 생성...")
    llm_with_tools = create_agent_llm(model_name="gpt-4o")
    print("✓ Agent LLM 준비 완료\n")

    # 3) Ground Truth 로드
    if not os.path.exists(GROUND_TRUTH_FILE):
        raise FileNotFoundError(GROUND_TRUTH_FILE)

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    print(f"[3단계] Ground Truth 로드 완료: 총 {len(gt_data)}개 샘플")
    samples = gt_data[:MAX_SAMPLES]
    print(f"→ 이번 실험에서 사용할 샘플 수: {len(samples)}\n")

    # 4) 샘플별 에이전트 실행
    print("[4단계] 샘플 처리 시작...\n")
    preds: List[Dict[str, Any]] = []

    for i, sample in enumerate(samples):
        text = sample["input_text"]
        print(f"--- [{i+1}/{len(samples)}] ---")
        print("입력 문장:", text, "\n")

        result = run_agent_on_sentence(llm_with_tools, text)
        preds.append(result)

    # 5) 메트릭 계산
    print("[5단계] 메트릭 계산 중...")
    compute_metrics(preds, samples)

    # 6) Conflict_DB 전체 요약
    print("[Conflict_DB 요약]")
    print(f"- 감지된 충돌 수: {len(conflict_db)}")
    for i, c in enumerate(conflict_db, start=1):
        print(f"  ({i}) [{'Hard' if c['conflict_type']=='Hard' else c['conflict_type']}] {c['reason']}")
        print(f"      evidence: {c['evidence']}")
        print(f"      문장: {c.get('text','(텍스트는 여기선 저장 안 함)')}\n")


if __name__ == "__main__":
    run_experiment()
