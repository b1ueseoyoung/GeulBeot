import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

# Retry Logic
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# ============================
# 0. 환경 설정
# ============================
os.environ['OPENAI_API_KEY'] = "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" 

STORY_FILE = "story.txt"
TEST_FILE = "ground_truth_100_v3.json"
VECTOR_DIR = "./chroma_story_rag_classified"

# ============================
# 1. AI 분석기
# ============================
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
def classify_chunk_with_retry(llm, text):
    msg = f"Classify (A=Fact, B=Emotion, C=Dialogue, D=Desc, P=Profile): {text[:200]}"
    return llm.invoke(msg).content.strip().upper()[:1]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def map_reduce_profiling(llm, full_text):
    print("   -> 🧠 [Map-Reduce] 캐릭터 프로파일링 중...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=30000, chunk_overlap=1000)
    large_chunks = splitter.split_text(full_text)
    
    partial_profiles = []
    for i, chunk in enumerate(large_chunks):
        map_prompt = f"등장인물 성격/특징 요약:\n{chunk}"
        res = llm.invoke(map_prompt).content
        partial_profiles.append(res)
        time.sleep(1)
        
    combined_text = "\n\n".join(partial_profiles)
    reduce_prompt = f"다음 인물 분석들을 하나로 통합하여 '이름: 성격' 형식으로 요약하라:\n{combined_text}"
    final_profile = llm.invoke(reduce_prompt).content
    return [line.strip() for line in final_profile.split('\n') if line.strip() and ':' in line]

# ============================
# 2. 시스템 상태 (버그 수정됨)
# ============================
class LoreKeeperState:
    def __init__(self):
        self.vectordb: Optional[Chroma] = None
        self.current_db: List[str] = []
        self.conflict_db: List[Dict] = []
        self.full_story_db: List[str] = [] # 🔥 [수정] 누락되었던 변수 추가!
        self.last_decision = None 

    def init_rag_system(self):
        if os.path.exists(VECTOR_DIR) and os.listdir(VECTOR_DIR):
            print(f"[System] 기존 DB 로드 ({VECTOR_DIR})")
            self.vectordb = Chroma(
                persist_directory=VECTOR_DIR, 
                embedding_function=OpenAIEmbeddings(),
                collection_name="classified_story_store"
            )
            return

        if not os.path.exists(STORY_FILE): return
        print("[System] 원고 처리 시작...")
        
        with open(STORY_FILE, "r", encoding="utf-8") as f:
            full_text = f.read()
            
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # 1. 프로파일링
        extracted_profiles = map_reduce_profiling(llm, full_text)
        profile_docs = extracted_profiles
        profile_metas = [{"category": "Profile", "chunk_id": -1} for _ in extracted_profiles]
        
        # 2. 청킹
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(full_text)
        
        story_metas = []
        for i, chunk in enumerate(chunks):
            try: cat = classify_chunk_with_retry(llm, chunk)
            except: cat = 'A'
            story_metas.append({"category": cat, "chunk_id": i})
            if (i+1)%50 == 0: print(f"      Classifying {i+1}...")

        # 3. 저장
        self.vectordb = Chroma.from_texts(
            texts=profile_docs + chunks,
            embedding=OpenAIEmbeddings(),
            metadatas=profile_metas + story_metas,
            collection_name="classified_story_store",
            persist_directory=VECTOR_DIR
        )
        print("[System] RAG 구축 완료!")

STATE = LoreKeeperState()

# ============================
# 3. Tools
# ============================
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_invoke(agent, messages):
    return agent.invoke({"messages": messages})

@tool
def search_lore_db(query: str) -> str:
    """[검색] 원작 내용을 검색합니다."""
    print(f"      🔎 [Search] '{query}'")
    if not STATE.vectordb: return "DB 없음"
    try: docs = STATE.vectordb.similarity_search(query, k=8)
    except: return "검색 에러"
    
    found_texts = []
    for d in docs:
        cat = d.metadata.get('category', '?')
        prefix = "★ [Profile]" if cat == "Profile" else f"[{cat} Type]"
        found_texts.append(f"{prefix} {d.page_content.replace(chr(10),' ')}")
        
    if not found_texts: return "검색 결과 없음."
    return "\n\n".join(found_texts)

@tool
def action_report_conflict(input_text: str, conflict_type: str, reason: str, evidence: str) -> str:
    """[행동] 충돌 신고"""
    STATE.conflict_db.append({"type": conflict_type, "text": input_text, "reason": reason, "evidence": evidence})
    STATE.last_decision = {"conflict": True, "reason": reason, "evidence": evidence}
    return "신고 완료"

@tool
def action_save_to_current_db(input_text: str, reason: str, evidence: str) -> str:
    """[행동] 승인"""
    STATE.current_db.append(input_text)
    STATE.last_decision = {"conflict": False, "reason": reason, "evidence": evidence}
    return "승인 완료"

# ============================
# 4. 에이전트 생성
# ============================
def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_lore_db, action_report_conflict, action_save_to_current_db]
    return create_react_agent(model=llm, tools=tools)

## ============================
# [수정] 5. 실행 파이프라인 (V19: 물리 법칙 우선주의)
# ============================
def run_evaluation():
    STATE.init_rag_system()
    agent = create_agent()

    # 🔥 [V19] 물리 법칙 최우선(Physics First) 프롬프트
    sys_msg = """
    너는 소설의 시공간적 모순을 잡아내는 'Logic Auditor'다.

    [절대 우선순위: 물리 법칙 > 캐릭터 성격]
    에이전트가 흔히 범하는 실수는 "성격이 맞으니까 물리적 오류도 눈감아주는 것"이다.
    **캐릭터의 성격이나 의도가 아무리 좋아도, 물리적으로 불가능하면 무조건 [Hard Conflict]다.**

    [검증 프로세스 (순서 엄수)]
    
    **STEP 1. 물리적 알리바이 검증 (Hard Conflict)**
    - 주어의 **'직전 위치(Last Known Location)'**를 검색하라.
    - 입력 문장의 이동이 **'물리적으로 가능한 시간/거리'**인지 계산하라.
    - 예시:
      - (DB) "만호는 대강당 연단에서 연설 중이다."
      - (입력) "만호는 곧바로 서울 친구를 찾아갔다."
      - (판정) **[충돌]** (대강당 -> 서울 순간이동 불가. '곧바로'라는 시간 부사가 모순됨.)
      - **주의:** 이 단계에서 걸리면 성격이고 뭐고 따지지 말고 즉시 신고하라.

    **STEP 2. 캐릭터성 검증 (Soft Conflict)**
    - STEP 1을 통과한 경우에만 성격을 따져라.
    - `★ [Profile]`과 비교하여 행동의 개연성을 판단하라.

    **STEP 3. 정보 부재 (Pass)**
    - 위치 정보도 없고, 성격 정보도 없으면 -> **[승인]** (작가의 새로운 설정)

    [행동 지침]
    - 핑계 대지 마라. "성격 정보를 몰라서 이동 판단 불가" 같은 헛소리는 하지 마라.
    - 위치 정보가 없으면 승인이지만, **위치 정보가 있는데 무시하면 치명적 오류다.**
    """

    if not os.path.exists(TEST_FILE):
        print("테스트 파일 없음")
        return
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n🚀 [V19 물리 법칙 우선 모드] 총 {len(data)}개 문장\n")
    
    score = {"total": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0}

    for idx, item in enumerate(data):
        text = item['input_text']
        gt_conflict = item['is_conflict']
        
        print(f"--- [Task {idx+1}] ---")
        
        # 지시사항: 위치 검색을 최우선으로 둠
        instruction = f"""
        검증 대상: '{text}'
        
        [지시]
        1. **[위치 검색]**: 주어가 직전에 어디 있었는지 `search_lore_db`로 확인하라. (쿼리 예: "만호의 현재 위치는?")
        2. **[물리 검증]**: 입력 문장의 이동/행동이 그 위치에서 가능한지 따져라.
        3. **[성격 검색]**: 물리적 문제가 없다면, 그때 성격을 검색해라.
        """
        
        try:
            safe_invoke(agent, [SystemMessage(content=sys_msg), HumanMessage(content=instruction)])
        except Exception as e:
            print(f"   ⚠️ 에러: {e}")

        decision = STATE.last_decision
        if decision:
            pred = decision['conflict']
            mark = "❌ 충돌" if pred else "✅ 승인"
            is_correct = (pred == gt_conflict)
            grade = "🙆‍♂️ 정답" if is_correct else "🤦‍♂️ 오답"
            print(f"   -> 판정: {mark} (정답: {'충돌' if gt_conflict else '통과'}) => {grade}")
            print(f"      🤔 이유: {decision.get('reason', 'N/A')}")
            ev = decision.get('evidence', 'N/A')
            print(f"      📝 근거: {ev[:80]}..." if len(ev) > 80 else f"      📝 근거: {ev}")
        else:
            print("   -> ☠️ 판단 실패")
            pred = False; is_correct = False
            
        score["total"] += 1
        if is_correct: score["correct"] += 1
        if pred and gt_conflict: score["tp"] += 1
        elif pred and not gt_conflict: score["fp"] += 1
        elif not pred and gt_conflict: score["fn"] += 1
        STATE.last_decision = None
        # time.sleep(0.5)

    print("="*60)
    acc = (score['correct'] / score['total']) * 100 if score['total'] > 0 else 0
    print(f"📈 최종 정확도: {acc:.1f}% ({score['correct']}/{score['total']})")
    
    if not STATE.conflict_db:
        STATE.full_story_db.extend(STATE.current_db)
        print(f"\n✨ 검증 완료! Full_Story_DB에 저장됨.")
    else:
        print(f"\n⚠️ {len(STATE.conflict_db)}건의 충돌 발생.")

if __name__ == "__main__":
    run_evaluation()