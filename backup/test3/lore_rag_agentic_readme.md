# Lore RAG Agentic 구조 변경 요약

## 1. 주요 변경사항

### 1-1. Tool 기반 구조 도입

`@tool` 데코레이터로 6개 Tool 정의:

1. `search_story_context`  
   - 역할: 스토리 맥락 검색 (Story/ Lore RAG 조회)

2. `get_current_db_settings`  
   - 역할: `Current_DB` 내 현재 설정/사실 조회

3. `classify_chunk_type`  
   - 역할: 청크 타입 분류  
   - 예: A(사실/설정), B(감정), C(대화), D(단순 서술)

4. `extract_facts_from_chunk`  
   - 역할: 청크에서 “사실”만 추출 (DB에 넣을 수 있는 형태로 정리)

5. `save_to_current_db`  
   - 역할: 추출된 사실을 `Current_DB`에 저장

6. `report_conflict_to_db`  
   - 역할: 검출된 충돌 내용을 `Conflict_DB`에 저장

---

### 1-2. AgentExecutor 사용

- `create_openai_tools_agent`  
  - LLM이 사용할 Tool 목록과 시스템 프롬프트를 묶어서 **에이전트** 생성
  - LLM이 **어떤 Tool을 언제 쓸지 스스로 결정**할 수 있도록 구성

- `AgentExecutor`  
  - LLM이 내린 결정에 따라 Tool 호출을 자동으로 실행
  - Tool 호출 결과를 다시 LLM에게 넘겨서 다음 Action을 결정하도록 함  
  - 즉, **Tool 호출 로직을 코드가 아니라 LangChain이 관리**

---

### 1-3. ReAct 패턴 적용

- ReAct: **Reasoning → Acting** 반복 구조
  - Reasoning: LLM이 현재 상황을 해석, “무엇을 해야 할지” 판단
  - Acting: 필요한 Tool을 선택해서 호출
- 각 단계에서 LLM이 자율적으로:
  - `classify_chunk_type` / `extract_facts_from_chunk` / `search_story_context` / `get_current_db_settings` / `save_to_current_db` / `report_conflict_to_db`
  - 중 어떤 Tool이 필요한지 판단하여 호출

---

### 1-4. 실행 흐름 (LLM에게 주는 지시)

LLM이 따라야 하는 기본 시나리오:

1. `classify_chunk_type` 로 현재 청크 타입 분류
2. 결과가 **D(단순 서술)** 이면:
   - 더 이상 처리하지 않고 **해당 청크는 스킵 후 종료**
3. 결과가 **A/B/C** 이면:
   - `extract_facts_from_chunk` 으로 사실(facts) 추출
4. 추출된 사실을 기준으로:
   - `search_story_context` 로 스토리 맥락 조회
   - `get_current_db_settings` 로 `Current_DB` 내 기존 설정 조회
5. 충돌 여부 판정 후:
   - **충돌이 있으면** → `report_conflict_to_db` 호출 (Conflict_DB 저장)
   - **충돌이 없으면** → `save_to_current_db` 호출 (Current_DB에 사실 저장)

---

## 2. 기존 구조와의 차이점

### 2-1. 기존: `lore_rag_test.py`

- 메서드 호출 순서가 **코드에 고정**
- 예:
  1. 분류 함수 호출
  2. RAG 검색
  3. 충돌 검사
  4. DB 저장
- 실행 흐름을 **파이썬 코드가 직접 제어**

### 2-2. 변경: `lore_rag_agentic.py`

- LLM이 **Tool을 선택하고 호출**
- `AgentExecutor` 가 Tool 호출을 자동 관리
- 실행 순서 예시는 “가이드”일 뿐, 실제로는:
  - LLM이 상황에 따라 “지금 어떤 Tool이 필요한지” 스스로 판단
  - 필요하면 일부 단계를 생략하거나, 반복 호출 가능
- 결과적으로 **진짜 에이전틱(Agentic) AI 구조**로 전환

---

## 3. 실행 방법

```bash
# 가상환경 활성화
cd GeulBeot
.venv\Scripts\Activate.ps1   # (Windows PowerShell 기준)

# 에이전트 실행
python lore_rag_agentic.py
