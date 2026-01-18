import os
import json
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

# ==========================
# 0. 기본 설정
# ==========================

# 환경변수에 OPENAI_API_KEY 미리 설정해 둔다고 가정 (export OPENAI_API_KEY=...)


# 1단계에서 생성한 리소스 경로
os.environ['OPENAI_API_KEY'] = "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" 
CHROMA_DIR = "./chroma_story_chunks"       # story.txt 전체 청킹 벡터 스토어
CHROMA_COLLECTION = "story_chunks"
CHUNK_META_FILE = "story_chunks_meta.json"  # 청크/타입 정보 (선택 사용)
GROUND_TRUTH_FILE = "ground_truth_100_v3.json"

# 실험용 샘플 개수 (필요하면 100으로 바꿔도 됨)
N_SAMPLES = 20

# in-memory "DB"
current_story_db: List[Dict[str, Any]] = []
conflict_db: List[Dict[str, Any]] = []

_vectordb: Optional[Chroma] = None
_chunk_meta_by_index: Dict[int, Dict[str, Any]] = {}


# ==========================
# 1. VectorStore & 메타 로드
# ==========================

def load_vectordb() -> Chroma:
    """1단계에서 만든 ChromaDB를 다시 로드한다."""
    global _vectordb
    if _vectordb is not None:
        return _vectordb

    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(f"Chroma 디렉토리를 찾을 수 없습니다: {CHROMA_DIR}")

    embeddings = OpenAIEmbeddings()
    _vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )
    return _vectordb


def load_chunk_meta():
    """story_chunks_meta.json을 로드해서 chunk_index → 메타 dict로 만든다."""
    global _chunk_meta_by_index
    if not os.path.exists(CHUNK_META_FILE):
        print(f"[경고] {CHUNK_META_FILE} 을 찾지 못했습니다. chunk_type 정보는 없이 진행합니다.")
        _chunk_meta_by_index = {}
        return

    with open(CHUNK_META_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 리스트 형태로 저장했다고 가정: [{"chunk_index": 0, "text": "...", "chunk_type": "A"}, ...]
    if isinstance(data, list):
        meta_list = data
    else:
        # 혹시 {"chunks": [...]} 형태라면
        meta_list = data.get("chunks", [])

    _chunk_meta_by_index = {m["chunk_index"]: m for m in meta_list}
    print(f"[메타 로드] story_chunks_meta.json 에서 {len(_chunk_meta_by_index)}개 청크 메타 로드")


# ==========================
# 2. Tools 정의
# ==========================

@tool
def search_story_context(query: str, k: int = 5) -> str:
    """
    [스토리 맥락 검색 도구]
    - story.txt 전체를 청킹한 ChromaDB에서 query와 비슷한 문맥을 k개 검색한다.
    - RAG 기반 내부 일관성 검사를 위해 항상 최소 1번 이상 사용하는 것을 권장한다.
    """
    vectordb = load_vectordb()
    docs = vectordb.similarity_search(query, k=k)

    lines = []
    for d in docs:
        idx = d.metadata.get("chunk_index")
        meta = _chunk_meta_by_index.get(idx, {})
        ctype = meta.get("chunk_type", "Unknown")
        lines.append(
            f"[chunk_index={idx}, type={ctype}]\n{d.page_content.strip()}"
        )

    if not lines:
        return "[검색 결과 없음]"

    return "[RAG 검색 결과]\n\n" + "\n\n---\n\n".join(lines)


@tool
def get_current_db_settings() -> str:
    """
    [Current_DB 조회 도구]
    지금까지 '충돌 없음'으로 판정되어 Current_DB에 저장된 설정들을 요약해서 보여준다.
    """
    if not current_story_db:
        return "현재까지 확정된 설정이 없습니다."

    lines = []
    for i, item in enumerate(current_story_db[-20:], start=1):
        note = item.get("note", "")
        sent = item.get("sentence", "")
        lines.append(f"{i}. {note}  (문장: {sent})")

    return "[Current_DB 최근 20개 설정]\n" + "\n".join(lines)


@tool
def save_to_current_db(sentence: str, note: str) -> str:
    """
    [정상 설정 저장 도구]
    - 충돌이 없다고 판단된 문장에 대한 요약/설정을 Current_DB에 저장한다.
    Args:
        sentence: 원본 입력 문장
        note: 이 문장에서 추출한 핵심 설정(자연어 요약)
    """
    current_story_db.append(
        {
            "sentence": sentence,
            "note": note,
        }
    )
    return "Current_DB에 설정이 저장되었습니다."


@tool
def report_conflict_to_db(
    sentence: str,
    conflict_type: str,
    reason: str,
    evidence: str,
) -> str:
    """
    [충돌 리포트 도구]
    - LLM이 판단한 충돌 정보를 Conflict_DB에 저장한다.
    Args:
        sentence: 충돌이 발생한 입력 문장
        conflict_type: 'Hard' 또는 'Soft'
        reason: 왜 충돌인지에 대한 설명
        evidence: RAG 검색 결과나 Current_DB에서 가져온 핵심 근거 문장
    """
    conflict_db.append(
        {
            "sentence": sentence,
            "conflict_type": conflict_type,
            "reason": reason,
            "evidence": evidence,
        }
    )
    return f"Conflict_DB에 {conflict_type} 충돌이 저장되었습니다."


TOOLS: List[BaseTool] = [
    search_story_context,
    get_current_db_settings,
    save_to_current_db,
    report_conflict_to_db,
]
TOOL_MAP: Dict[str, BaseTool] = {t.name: t for t in TOOLS}


# ==========================
# 3. Agent LLM + tool-calling 루프
# ==========================

SYSTEM_PROMPT = """
너는 웹소설 설정을 관리하는 자율형 에이전트 'Lore Keeper'다.

너의 임무:
- 한 문장이 기존 스토리(story.txt 전체, ChromaDB로 제공됨) 및 Current_DB에 저장된 설정들과
  논리적으로 충돌하는지 검사한다.

사용 가능한 도구:
1) search_story_context(query, k=5)
   - story.txt 전체를 벡터 검색해서, 이 문장과 관련된 맥락을 찾아온다.
2) get_current_db_settings()
   - 지금까지 '충돌 없음'으로 기록된 설정 요약들을 가져온다.
3) save_to_current_db(sentence, note)
   - 충돌이 없다고 판단된 문장에 대한 핵심 설정 요약을 저장한다.
4) report_conflict_to_db(sentence, conflict_type, reason, evidence)
   - 충돌이라고 판단된 문장을 Conflict_DB에 기록한다.

작업 절차(반드시 이 순서를 따르려고 노력해라):
1. 먼저, 이 문장에서 중요한 인물/장소/시간/사건 정보를 한두 줄로 자연어로 정리한다.
2. 이 요약을 query로 사용해 search_story_context(...)를 반드시 **최소 1회 이상** 호출하라.
3. 필요하다면 get_current_db_settings()도 호출해라.
4. 검색된 RAG 맥락 + Current_DB 설정을 바탕으로 충돌 여부를 판단하라.
   - Hard Conflict:
     · 이미 죽은 인물이 다시 살아 등장
     · 시간/공간/인과관계 상 절대 일어날 수 없는 사건
     · 세계관의 핵심 규칙을 정면으로 위반
   - Soft Conflict:
     · 캐릭터 성격, 말투, 관계, 감정 흐름이 기존 묘사와 현저히 어긋남
5. 충돌이 없다면:
   - 이 문장의 핵심 설정을 1~2문장으로 요약하고
   - save_to_current_db(sentence, note)를 호출하라.
6. 충돌이 있다면:
   - conflict_type을 'Hard' 또는 'Soft' 중 하나로 정하고
   - reason에는 왜 충돌인지 한국어로 설명하고
   - evidence에는 search_story_context나 Current_DB에서 찾은 핵심 근거 문장을 넣어
   - report_conflict_to_db(...)를 반드시 호출하라.

최종 응답 형식:
- 모든 tool 호출이 끝난 후, 마지막에는 **오직 JSON만** 한 번 출력해야 한다.
- JSON 키:
  - is_conflict: true 또는 false
  - conflict_type: "Hard", "Soft", "None" 중 하나
  - reason: 한국어 설명 문자열
  - evidence: 근거가 된 스토리/설정 요약 (한두 문장)
다른 설명 문장, 코드블록, 마크다운은 절대 넣지 말고 JSON만 출력해라.
""".strip()


def create_agent_llm(model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    """tool-calling이 가능한 LLM 생성"""
    llm = ChatOpenAI(model=model_name, temperature=0)
    return llm.bind_tools(TOOLS)


def run_agent_on_sentence(llm_with_tools: ChatOpenAI, sentence: str) -> Dict[str, Any]:
    """
    한 문장에 대해:
    - 시스템 프롬프트 + HumanMessage로 시작
    - LLM/tool-calling 루프 수행
    - 마지막 assistant 메시지(JSON)를 파싱해서 return
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"검사할 문장:\n{sentence.strip()}"),
    ]

    last_ai = None

    for step in range(6):  # safety: 최대 6스텝
        ai_msg = llm_with_tools.invoke(messages)
        last_ai = ai_msg
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None)
        if tool_calls:
            # 각 tool call 실행 후 ToolMessage 추가
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("args", {}) or {}
                tool_obj = TOOL_MAP[name]
                # langchain tool은 .invoke(dict)로 호출
                result = tool_obj.invoke(args)
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
            continue
        else:
            break

    if last_ai is None:
        raise RuntimeError("LLM 응답이 없습니다.")

    # 마지막 assistant 메시지에서 JSON 파싱
    raw = last_ai.content
    try:
        data = json.loads(raw)
    except Exception:
        # 혹시 주변에 다른 텍스트가 있으면 대괄호/중괄호 부분만 추출 시도
        raw_stripped = raw.strip()
        start = raw_stripped.find("{")
        end = raw_stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw_stripped[start : end + 1])
            except Exception:
                data = {
                    "is_conflict": False,
                    "conflict_type": "None",
                    "reason": f"JSON 파싱 실패: {raw_stripped[:100]}...",
                    "evidence": "",
                }
        else:
            data = {
                "is_conflict": False,
                "conflict_type": "None",
                "reason": f"JSON 형식이 아님: {raw_stripped[:100]}...",
                "evidence": "",
            }

    # 안전하게 기본값 채우기
    return {
        "is_conflict": bool(data.get("is_conflict", False)),
        "conflict_type": data.get("conflict_type", "None"),
        "reason": data.get("reason", ""),
        "evidence": data.get("evidence", ""),
        "raw": raw,
    }


# ==========================
# 4. 메트릭 계산
# ==========================

def compute_metrics(preds: List[Dict[str, Any]], gts: List[Dict[str, Any]]):
    assert len(preds) == len(gts)

    def norm_type(x: str) -> str:
        s = (x or "").lower()
        if "hard" in s:
            return "Hard"
        if "soft" in s:
            return "Soft"
        return "None"

    N = len(preds)
    correct_conf = 0
    tp = fp = fn = 0
    correct_type = 0
    true_conf_cnt = 0

    for p, g in zip(preds, gts):
        y_true = bool(g.get("is_conflict", False))
        y_pred = bool(p.get("is_conflict", False))

        if y_true == y_pred:
            correct_conf += 1

        if y_pred and y_true:
            tp += 1
        elif y_pred and not y_true:
            fp += 1
        elif not y_pred and y_true:
            fn += 1

        if y_true:
            true_conf_cnt += 1
            gt_t = norm_type(g.get("conflict_type", "None"))
            pr_t = norm_type(p.get("conflict_type", "None"))
            if gt_t in ["Hard", "Soft"] and pr_t == gt_t:
                correct_type += 1

    acc_conf = correct_conf / N if N > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    type_acc = correct_type / true_conf_cnt if true_conf_cnt > 0 else 0.0

    print("\n================ [실험 결과 요약] ================")
    print(f"- 샘플 수: {N}")
    print(f"- Conflict 여부 정확도: {acc_conf:.3f}")
    print(f"- Precision (conflict): {precision:.3f}")
    print(f"- Recall (conflict):    {recall:.3f}")
    print(f"- F1 (conflict):        {f1:.3f}")
    print(f"- Hard/Soft 타입 정확도(충돌인 샘플 기준): {type_acc:.3f}")
    print("=================================================\n")


# ==========================
# 5. 전체 실험
# ==========================

def run_experiment():
    print("🤖 에이전틱 Lore Keeper + RAG 실험 (2단계) 시작...\n")

    # 1) VectorStore + 메타 로드
    print("[1단계] ChromaDB + 메타 로드 중...")
    load_vectordb()
    load_chunk_meta()
    print("✓ VectorStore & 메타 준비 완료\n")

    # 2) Agent LLM 생성
    print("[2단계] Agent LLM 생성 중...")
    agent_llm = create_agent_llm()
    print("✓ Agent LLM 준비 완료\n")

    # 3) Ground Truth 로드
    if not os.path.exists(GROUND_TRUTH_FILE):
        raise FileNotFoundError(f"{GROUND_TRUTH_FILE} 을 찾을 수 없습니다.")
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    samples = gt_data[:N_SAMPLES]
    print(f"[3단계] Ground Truth 로드 완료: {len(samples)}개 샘플 사용\n")

    # 4) 샘플 처리
    print("[4단계] 샘플 처리 시작...\n")
    preds: List[Dict[str, Any]] = []

    for i, sample in enumerate(samples, start=1):
        sentence = sample["input_text"]
        print(f"--- [{i}/{len(samples)}] ---")
        print("입력 문장:", sentence, "\n")

        result = run_agent_on_sentence(agent_llm, sentence)
        preds.append(result)

    # 5) 메트릭 계산
    print("[5단계] 메트릭 계산 중...")
    compute_metrics(preds, samples)

    # 6) Conflict_DB / Current_DB 요약
    print("[Conflict_DB 요약]")
    print(f"- 감지된 충돌 수: {len(conflict_db)}")
    for i, c in enumerate(conflict_db, start=1):
        print(f"  ({i}) [{c.get('conflict_type')}] {c.get('reason')}")
        print(f"      문장: {c.get('sentence')}")
        print(f"      evidence: {c.get('evidence')}\n")

    print("[Current_DB 요약]")
    print(f"- 저장된 설정 수: {len(current_story_db)}")
    for i, item in enumerate(current_story_db, start=1):
        print(f"  ({i}) note: {item.get('note')}")
        print(f"      문장: {item.get('sentence')}\n")


if __name__ == "__main__":
    run_experiment()
