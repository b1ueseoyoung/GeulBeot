# main.py 코드 리뷰

## 환경 설정 (가상환경)
```
pip install pandas langchain-openai langgraph faiss-cpu
```

## API
```
OPENAI_API_KEY = "your_openai_api_key" <br>
```

## 실행
```
python main.py
```

## DB / 파일 입출력
| 이름 | 설명 | 형식 |
| --- | --- | --- |
| Lore_DB | 확정된 설정 데이터 | jsonl 파일 + 벡터 DB(FAISS) |
| Full_Story_DB | 소설 원문 청크 데이터 | jsonl 파일 + 벡터 DB(FAISS) |
| Conflict_DB | 충돌 로그 | jsonl 파일 |
| Current_DB | 현재 처리 중인 회차의 설정 데이터 | 단순 List + 임시 벡터 DB(FAISS) |


### 관련 함수

- **`_normalize_enum`**
  - 입력값이 정해진 Enum 범위 내에 있는지 확인 후 대문자로 통일
  - 범위 밖일 경우 기본값 할당

- **`_build_lore_item`**
  - LLM을 통해 추출한 Fact를 `Lore_DB` 스키마에 맞춰 변환

- **`_load_dataframe` / `_write_dataframe`**
  - `jsonl` 형식 파일을 pandas `DataFrame`으로 가져오기/저장하기
  - 쉬운 파일 입출력 관리 목적

- **`load_persistent_state`**
  - `main.py` 실행 시 기존에 저장된 모든 데이터(DB) 및 벡터 DB를 메모리에 적재

- **`_extract_from_messages`**
  - LangGraph 실행 결과의 `ToolMessage`에서 아래 정보를 추출
    - 추출된 Fact
    - 충돌 판정 결과
    - 연관된 소설 원문

### 벡터 스토어

**Lore_DB / Full_Story_DB (영구 저장)**

- **`get_lore_db_vectordb` / `get_full_story_db_vectordb`**
  - 저장된 `Lore_DB`, `Full_Story_DB` 벡터 스토어 로드
  - 파일이 없으면 `None` 반환

- **`add_to_lore_db_vectorstore` / `add_to_full_story_db_vectorstore`**
  - 추출된 설정과 원문 청크를 각각 벡터화하여 저장

- **`save_lore_db_to_file` / `save_full_story_to_file`**
  - 추출된 설정과 원문 청크를 `jsonl` 파일로 저장


**Current (회차 단위 임시 저장)**

- **`get_current_chunk_vectordb` / `add_to_current_chunk_vectorstore`**
  - 현재 읽고 있는 문장(충돌 검사를 진행할 청크)을 즉시 벡터화
  - 한 회차 내 설정 충돌 검사를 위해 사용

- **`get_current_story_vectordb` / `add_to_current_db_vectorstore`**
  - 현재 회차에서 확정된 설정(`Lore_Items`)을 임시 벡터에 저장
  - 한 회차 내 설정 충돌 검사를 위해 사용

- **`save_to_current_db`**
  - 충돌 없는 설정을 `Current_DB`에 저장

- **`reset_current_episode_state`**
  - 현재 회차의 모든 임시 데이터를 리셋


**Conflict (충돌 기록)**

- **`report_conflict_to_db`**
  - 설정 충돌 로그를 `Conflict_DB` 리스트에 추가(append)
  - 기록 항목 예시:
    - `conflict_type`
    - `evidence`
    - `facts`
    - `conflicting_text`

- **`append_conflicts_to_file`**
  - `Conflict_DB`를 파일로 저장

<br>

### Agent Tools

에이전트(LLM)가 스스로 판단하여 아래 도구 목록 중 필요한 도구를 호출한다.

- **`search_current_db`**
  - 현재 회차의 임시 저장된 설정들 중, 현재 청크와 연관된 설정 검색(RAG)
  - 추출된 Facts(=`lore_items`)를 벡터 검색

- **`search_lore_db`**
  - 이전 회차의 저장된 설정들 중, 현재 청크와 연관된 설정 검색(RAG)
  - 추출된 Facts(=`lore_items`)를 벡터 검색

- **`search_full_story_db`**
  - 이전 회차의 원문 소설 청크 데이터를 벡터 검색(RAG)

- **`search_current_chunks`**
  - 현재 회차의 원문 소설 청크 데이터를 벡터 검색(RAG)

- **`classify_chunk_type`**
  - 현재 입력된 청크(소설 일부)를 유형으로 분류.
    - **A**: 사실/설정
    - **B**: 감정/내면
    - **C**: 대화
    - **D**: 단순 서술
  - **D로 분류된 데이터는 설정 추출을 하지 않기 위한 조치**

- **`extract_facts_from_chunk`**
  - 청크 데이터에서 설정(`lore_items`)을 추출.
  - 구성 필드:
    - `item_type`, `category`, `target_group`, `chunk_type`, `subject`, `condition`, `effect`, `text`

- **`judge_conflict`**
  - 아래 검색 결과들을 비교해 설정 충돌 여부를 판단.
    - `search_current_db`
    - `search_lore_db`
    - `search_full_story_db`
    - `search_current_chunks`

<br>

### Agent(LLM)

에이전트(LLM)를 생성하고 호출한다.

- **`create_lore_keeper_agent`**
  - LLM 모델 설정 + Agent가 호출할 수 있는 도구(tools) + 기본 작동 흐름(workflow + system prompt) 작성

- **`process_chunk_with_agent`**
  - 에이전트(LLM)로 청크 하나를 처리.
    - `Current_DB`에 저장하거나
    - `Conflict_DB`에 충돌 리포트를 추가.
  - 기본적으로 `is_conflict = False`
  - `judge_result`가 존재(충돌 판단)하더라도:
    - `conflicting_text`(충돌 증거 원문)가 **실제로 없으면** `is_conflict = False`
    - `conflicting_text`가 **실제로 존재할 때만** `is_conflict = True`

<br>

### 실행

- **`run_manual_episode`**
  - 한 회차에 대한 에이전트를 통한 설정 충돌 검사 진행

- **`run_multiple_episodes`**
  - 테스트 용이성을 위해 다회차를 한 번에 검사
  - 루프를 돌며 `run_manual_episode` 호출

- **`__main__`**
  - 텍스트 청킹 후 `run_multiple_episodes` 호출


