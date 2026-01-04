"""
진짜 에이전틱 AI 구조로 구현한 Lore Keeper
- LangChain AgentExecutor 사용
- LLM이 스스로 tool을 선택하고 호출
- ReAct 패턴 (Reasoning + Acting)
"""

import os
import json
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.tools import tool
# from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.tools import tool
from langchain.agents import create_agent

# ========= 경로 설정 =========
STORY_FILE = "story.txt"
GROUND_TRUTH_FILE = "ground_truth_100_v3.json"
VECTOR_DIR = "./chroma_story_rag"


# ========= 전역 변수 (in-memory DB) =========
current_story_db: List[Dict[str, Any]] = []
conflict_db: List[Dict[str, Any]] = []
_vectordb: Optional[Chroma] = None  # 전역 벡터 스토어


# ========= 0. VectorStore 준비 =========
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
        length_function=len
    )
    chunks = splitter.split_text(story_text)

    print(f"[VectorStore] story.txt에서 {len(chunks)}개의 청크 생성")
    
    if os.path.exists(VECTOR_DIR):
        print(f"[VectorStore] 기존 벡터 스토어 발견: {VECTOR_DIR}")
        embeddings = OpenAIEmbeddings()
        _vectordb = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embeddings,
            collection_name="story_chunks",
        )
        print("[VectorStore] 기존 벡터 스토어 로드 완료")
    else:
        print(f"[VectorStore] Embeddings 생성 중...")
        embeddings = OpenAIEmbeddings()
        print(f"[VectorStore] Chroma 벡터 스토어 생성 중... ({len(chunks)}개 청크를 벡터화합니다)")
        _vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=[{"chunk_index": i} for i in range(len(chunks))],
            persist_directory=VECTOR_DIR,
            collection_name="story_chunks",
        )
        print("[VectorStore] story_chunks 컬렉션 생성 완료")
    
    return _vectordb


# ========= 1. Tools 정의 (LLM이 호출할 수 있는 도구들) =========

@tool
def search_story_context(query: str) -> str:
    """
    [스토리 맥락 검색 도구]
    VectorStore에서 주어진 쿼리와 관련된 스토리 맥락을 검색합니다.
    충돌 검사를 위해 기존 스토리 내용을 찾을 때 사용하세요.
    
    Args:
        query: 검색할 키워드나 문장
        
    Returns:
        관련된 스토리 맥락 텍스트
    """
    vectordb = build_story_vectordb()
    docs = vectordb.similarity_search(query, k=5)
    context = "\n\n".join([d.page_content for d in docs])
    return f"[스토리 맥락 검색 결과]\n{context}" if context else "[관련 스토리 맥락 없음]"


@tool
def get_current_db_settings() -> str:
    """
    [Current_DB 조회 도구]
    현재까지 확정된 설정들(Current_DB)을 조회합니다.
    충돌 검사를 위해 이전에 저장된 설정들을 확인할 때 사용하세요.
    
    Returns:
        Current_DB에 저장된 설정들의 텍스트 요약
    """
    if not current_story_db:
        return "현재 회차에서 확정된 설정이 아직 없습니다."
    
    lines = []
    for item in current_story_db[-20:]:  # 최근 20개만
        fact = item.get("fact", "")
        text = item.get("text", "")
        if isinstance(fact, dict):
            fact_str = json.dumps(fact, ensure_ascii=False)
        else:
            fact_str = str(fact)
        lines.append(f"- {fact_str} (원문: {text})")
    
    return "[Current_DB에 저장된 설정들]\n" + "\n".join(lines)


@tool
def classify_chunk_type(chunk: str) -> str:
    """
    [청크 타입 분류 도구]
    주어진 문장을 A/B/C/D로 분류합니다.
    
    - A: 사실/설정 (세계관 규칙, 인물/아이템의 상태, 사건, 배경 설정 등)
    - B: 감정/내면 (인물의 감정, 심리, 내면 독백 등)
    - C: 대화 (직접 화법 대사)
    - D: 단순 서술 (설정과 크게 상관없는 묘사, 분위기 설명 등)
    
    Args:
        chunk: 분류할 문장
        
    Returns:
        'A', 'B', 'C', 또는 'D'
    """
    # 이 tool은 LLM이 직접 판단하도록 하거나, 별도 LLM 호출로 처리
    # 여기서는 간단히 LLM을 호출해서 분류
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    system_msg = (
        "너는 웹소설 설정을 관리하는 에이전트 AI 'Lore Keeper'야.\n"
        "아래에 주어지는 문장을 다음 네 가지 중 하나로 분류해.\n"
        "A: 사실/설정 (세계관 규칙, 인물/아이템의 상태, 사건, 배경 설정 등)\n"
        "B: 감정/내면 (인물의 감정, 심리, 내면 독백 등)\n"
        "C: 대화 (직접 화법 대사)\n"
        "D: 단순 서술 (설정과 크게 상관없는 묘사, 분위기 설명 등)\n"
        "반드시 한 글자(A/B/C/D)만 출력해."
    )
    
    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"다음 문장을 분류해줘:\n{chunk}"}
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
    주어진 문장에서 설정 충돌 검사를 위해 비교 가능한 사실들을 JSON 배열로 추출합니다.
    
    Args:
        chunk: 분석할 문장
        chunk_type: 청크 타입 ('A', 'B', 'C', 또는 'D')
        
    Returns:
        JSON 배열 형식의 사실들 (각 사실은 subject, predicate, obj, category, time, location 필드를 가짐)
    """
    if chunk_type == "D":
        return "[]"
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    system_msg = (
        "너는 웹소설 설정 관리용 분석 에이전트야.\n"
        "주어진 문장을 보고, 설정 충돌 검사를 위해 비교 가능한 사실들을 JSON 배열로 추출해.\n"
        "각 사실은 다음 필드를 가진 객체야:\n"
        "- subject: 인물/대상 이름\n"
        "- predicate: 행동/상태/관계 등에 대한 서술\n"
        "- obj: 대상이 있는 경우 그 대상 (없으면 빈 문자열)\n"
        "- category: FACT / EMOTION / DIALOGUE 중 하나\n"
        "- time: 시간 정보가 있으면 자연어로 (없으면 빈 문자열)\n"
        "- location: 장소 정보가 있으면 자연어로 (없으면 빈 문자열)\n"
        "반드시 JSON 배열만 출력해. 다른 말은 쓰지 마."
    )
    
    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"다음 문장에서 사실을 추출해줘:\n{chunk}"}
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
    
    Args:
        chunk: 원본 문장
        facts_json: 추출된 사실들의 JSON 배열 문자열
        
    Returns:
        저장 완료 메시지
    """
    try:
        facts = json.loads(facts_json) if facts_json else []
        for fact in facts:
            current_story_db.append({
                "text": chunk,
                "fact": fact,
            })
        return f"Current_DB에 {len(facts)}개의 사실이 저장되었습니다."
    except Exception as e:
        return f"저장 실패: {str(e)}"


@tool
def report_conflict_to_db(chunk: str, conflict_type: str, reason: str, facts_json: str = "[]") -> str:
    """
    [충돌 리포트 도구]
    감지된 설정 충돌을 Conflict_DB에 저장합니다.
    
    Args:
        chunk: 충돌이 발생한 문장
        conflict_type: 'Hard' 또는 'Soft'
        reason: 충돌 이유 설명
        facts_json: 관련 사실들의 JSON 배열 (선택사항)
        
    Returns:
        저장 완료 메시지
    """
    try:
        facts = json.loads(facts_json) if facts_json else []
        conflict_db.append({
            "text": chunk,
            "conflict_type": conflict_type,
            "reason": reason,
            "facts": facts,
        })
        return f"Conflict_DB에 {conflict_type} 충돌이 저장되었습니다."
    except Exception as e:
        return f"저장 실패: {str(e)}"


# ========= 2. 에이전트 생성 =========

def create_lore_keeper_agent(model_name: str = "gpt-4o"):
    """
    Lore Keeper 에이전트를 생성합니다.
    LangGraph의 create_react_agent를 사용하여 LLM이 스스로 tool을 선택하고 호출합니다.
    """
    # Tool 리스트
    tools = [
        search_story_context,
        get_current_db_settings,
        classify_chunk_type,
        extract_facts_from_chunk,
        save_to_current_db,
        report_conflict_to_db,
    ]
    
    # LLM 초기화
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    # 시스템 프롬프트
    system_prompt = """너는 웹소설 설정을 관리하는 자율형 AI 에이전트 'Lore Keeper'야.

너의 임무는 주어진 문장을 분석해서 설정 충돌을 검사하는 거야.

**작업 절차:**
1. 먼저 classify_chunk_type으로 문장을 A/B/C/D로 분류해
2. D 타입이면 단순 서술이므로 PASS (작업 종료)
3. A/B/C 타입이면:
   a. extract_facts_from_chunk로 사실들을 추출해
   b. search_story_context와 get_current_db_settings로 관련 설정을 조회해
   c. 추출한 사실과 기존 설정을 비교해서 충돌 여부를 판단해
   d. 충돌이 있으면 report_conflict_to_db로 저장하고, 없으면 save_to_current_db로 저장해

**충돌 판정 기준:**
- Hard Conflict: 이미 죽은 인물이 다시 살아서 등장, 세계관 핵심 규칙 위반, 시간적으로 불가능한 사건
- Soft Conflict: 캐릭터 성격/말투/관계가 이전과 너무 다름, 감정 흐름이 뜬금없음

**중요:**
- 각 단계마다 필요한 tool을 스스로 선택해서 호출해
- 충돌 판정은 추출한 사실과 조회한 설정을 비교해서 논리적으로 판단해
- 판단 근거를 명확히 설명해

지금부터 주어진 문장을 분석해봐."""

    # LangChain의 create_agent 사용 (LangGraph의 create_react_agent는 deprecated)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    
    return agent


# ========= 3. 청크 처리 함수 =========

def process_chunk_with_agent(agent, chunk: str, index: int) -> Dict[str, Any]:
    """
    에이전트를 사용해서 청크를 처리합니다.
    LLM이 스스로 판단해서 필요한 tool을 호출합니다.
    """
    print(f"\n--- [Chunk {index+1}] ---")
    print("입력 문장:", chunk)
    print("\n[에이전트 실행 시작]")
    print("=" * 60)
    
    # 에이전트에게 작업 지시
    input_text = f"다음 문장을 분석해서 설정 충돌을 검사해줘:\n\n{chunk}"
    
    try:
        # LangGraph의 invoke는 messages를 받습니다
        from langchain_core.messages import HumanMessage
        result = agent.invoke({"messages": [HumanMessage(content=input_text)]})
        
        # LangGraph의 결과는 messages 리스트를 반환합니다
        messages = result.get("messages", [])
        if messages:
            # 마지막 메시지가 최종 응답
            output = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        else:
            output = str(result)
        
        print("=" * 60)
        print("[에이전트 실행 완료]")
        print(f"결과: {output[:200]}...")  # 처음 200자만 출력
        
        # 결과 파싱 시도
        chunk_type = "Unknown"
        is_conflict = False
        conflict_type = "None"
        
        # 출력에서 정보 추출 시도
        if "D" in output or "단순 서술" in output:
            chunk_type = "D"
        elif "A" in output or "사실/설정" in output:
            chunk_type = "A"
        elif "B" in output or "감정" in output:
            chunk_type = "B"
        elif "C" in output or "대화" in output:
            chunk_type = "C"
        
        if "Hard" in output or "하드" in output:
            is_conflict = True
            conflict_type = "Hard"
        elif "Soft" in output or "소프트" in output:
            is_conflict = True
            conflict_type = "Soft"
        
        return {
            "chunk_type": chunk_type,
            "is_conflict": is_conflict,
            "conflict_type": conflict_type,
            "agent_output": output,
        }
        
    except Exception as e:
        print(f"[에러 발생] {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "chunk_type": "Error",
            "is_conflict": False,
            "conflict_type": "None",
            "error": str(e),
        }


# ========= 4. 평가 메트릭 계산 =========

def compute_metrics(results: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]):
    """ground_truth와 비교해서 메트릭 계산"""
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


# ========= 5. 전체 실험 실행 =========

def run_experiment():
    """전체 실험 실행"""
    print("🤖 진짜 에이전틱 AI Lore Keeper 실험 시작...\n")
    print("=" * 60)
    print("이번에는 LLM이 스스로 tool을 선택하고 호출합니다!")
    print("=" * 60)
    
    # 1) 벡터 스토어 준비
    print("\n[1단계] 벡터 스토어 준비 중...")
    build_story_vectordb()
    print("✓ 완료\n")
    
    # 2) 에이전트 생성
    print("[2단계] 에이전트 생성 중...")
    agent = create_lore_keeper_agent()
    print("✓ 완료\n")
    
    # 3) Ground Truth 로드
    if not os.path.exists(GROUND_TRUTH_FILE):
        raise FileNotFoundError(f"ground_truth 파일을 찾을 수 없습니다: {GROUND_TRUTH_FILE}")
    
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    
    print(f"[3단계] Ground Truth 로드 완료: {len(gt_data)}개 샘플\n")
    
    # 4) 각 샘플 처리
    print("[4단계] 에이전트로 샘플 처리 시작...\n")
    results: List[Dict[str, Any]] = []
    
    # 테스트를 위해 처음 5개만 처리 (전체 처리하려면 주석 해제)
    test_samples = gt_data[:5]  # 처음 5개만
    # test_samples = gt_data  # 전체 처리
    
    for i, sample in enumerate(test_samples):
        text = sample["input_text"]
        result = process_chunk_with_agent(agent, text, index=i)
        results.append(result)
    
    # 5) 메트릭 계산
    print("\n[5단계] 메트릭 계산 중...")
    compute_metrics(results, test_samples)
    
    # 6) 결과 요약
    print("[Conflict_DB 요약]")
    print(f"- 감지된 충돌 수: {len(conflict_db)}")
    for i, c in enumerate(conflict_db[:10], start=1):
        print(f"  ({i}) [{c.get('conflict_type', 'Unknown')}] {c.get('reason', '')}")
        print(f"      문장: {c.get('text', '')}\n")
    
    print("[Current_DB 요약]")
    print(f"- 저장된 설정 수: {len(current_story_db)}")
    for i, item in enumerate(current_story_db[:10], start=1):
        fact = item.get("fact", {})
        print(f"  ({i}) fact: {fact}")
        print(f"      문장: {item.get('text', '')}\n")


# ========= 6. main =========
if __name__ == "__main__":
    run_experiment()

