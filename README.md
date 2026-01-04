## main.py 코드 리뷰
---
### 환경 설정 (가상환경)
```
pip install pandas langchain-openai langgraph faiss-cpu
```

### API
```
OPENAI_API_KEY = "your_openai_api_key" <br>
```

### 실행
```
python main.py
```

### DB / 파일 입출력
| 이름 | 설명 | 형식 |
| --- | --- | --- |
| Lore_DB | 확정된 설정 데이터 | jsonl 파일 + 벡터 DB(FAISS) |
| Full_Story_DB | 소설 원문 청크 데이터 | jsonl 파일 + 벡터 DB(FAISS) |
| Conflict_DB | 충돌 로그 | jsonl 파일 |
| Current_DB | 현재 처리 중인 회차의 설정 데이터 | 단순 List + 임시 벡터 DB(FAISS) |
<br>

#### 관련 함수
- **_normalize_enum** : 입력값이 정해진 Enum 범위 내에 있는지 확인 후 대문자로 통일. 만약 범위 밖일 경우 기본값 할당.
- **_build_lore_item** : LLM을 통해 추출한 Fact를 Lore_DB 스키마에 맞춰 변환.
- **_load_dataframe** / **_write_dataframe** : jsonl 형식의 파일을 pandas로 dataframe 형식으로 가져오기/저장하기. 쉬운 파일 입출력 관리 목적.
- **load_persistent_state** : main.py 실행 시 기존에 저장된 모든 데이터(DB) 및 벡터 DB 메모리에 적재.
- **_extract_from_messages** : LangGraph 실행 결과의 ToolMessage에서 추출된 Fact, 충돌 판정 결과, 연관된 소설 원문을 추출.

**벡터 스토어**
- **get_lore_db_vectordb** / **get_full_story_db_vectordb** : 저장된 Lore_DB, Full_Story_Db 벡터 스토어를 로드. 파일이 없으면 None 반환.
- **add_to_lore_db_vectorstore** / **add_to_full_story_db_vectorstore** : 추출된 설정과 원문 청크를 각각 벡터화하여 저장.
- **save_lore_db_to_file** / **save_full_story_to_file** : 추출된 설정과 원문 청크를 jsonl 파일로 저장.


- **get_current_chunk_vectordb** / **add_to_current_chunk_vectorstore** : 현재 읽고 있는 문장(충돌 검사를 진행할 청크)를 즉시 벡터화. 한 회차 내에서 설정 충돌 검사를 위해 사용.
- **get_current_story_vectordb** / **add_to_current_db_vectorstore** : 현재 회차에서 확정된 설정(Lore_Items)을 임시 벡터에 저장. 한 회차 내에서 설정 충돌 검사를 위해 사용.
- **save_to_current_db** : 충돌 없는 설정을 Current_DB에 저장.
- **reset_current_episode_state** : 현재 회차의 모든 임시 데이터를 리셋.

- **report_conflict_to_db** : 설정 충돌 로그를 Conflict_DB List에 추가(append). conflict_type, evidence, facts, conflicting_text 등이 기록됨.
- **append_conflicts_to_file** : Conflict_DB를 파일로 저장.


### Agent Tools
에이전트(LLM)이 스스로 생각하여, 아래 도구 목록 중 필요한 것으로 판단되는 도구를 호출한다.
- **search_current_db** : 현재 회차의 임시 저장된 설정들 중 현재 청크와 연관된 설정 검색(RAG). 추출된 Facts(=lore_items)를 벡터 검색.
- **search_lore_db** : 이전 회차의 저장된 설정들 중 현재 청크와 연관된 설정 검색(RAG). 추출된 Facts(=lore_items)를 벡터 검색.
- **search_full_story_db** : 이전 회차의 원문 소설 청크 데이터를 벡터 검색(RAG).
- **search_current_chunks** : 현재 회차의 원문 소설 청크 데이터를 벡터 검색(RAG).
- **classify_chunk_type** : 현재 입력된 청크 데이터(소설 일부)가 각각 A: 사실/설정, B: 감정/내면, C: 대화, D: 단순 서술 중 어디에 해당하는 지 분류. 이후 D로 분류된 데이터에 대한 설정 추출을 하지 않기 위한 조치.
- **extract_facts_from_chunk** : 청크 데이터에서 설정(lore_items)을 추출. item_type, category, target_group, chunk_type, subject, condition, effect, text로 구성됨.
- **judge_conflict** : search_current_db, search_lore_db, search_full_story_db, search_current_chunks를 통해 검색된 관련 설정과 비교하여 설정 충돌이 존재하는지 여부 판단.


### Agent(LLM)
에이전트(LLM)를 생성하고 호출한다.
- **create_lore_keeper_agent** : LLM 모델을 비롯해 Agent가 호출할 수 있는 도구(tools)와 기본 작동 흐름(workflow + system prompt)을 작성.
- **process_chunk_with_agent** : 에이전트(LLM)로 청크 하나를 처리하고, Current_DB에 저장하거나 Conflict_DB에 충돌 리포트를 추가. 기본적으로 is_conflict=False. 만약 충돌(judge_result가 존재)이 있지만, conflicting_text(충돌의 증거가 되는 소설 원문)가 실제로 존재하지 않을 경우 is_conflict=False 설정. 만약 conflicting_text가 실제로 존재할 경우에만 is_conflict=True로 설정.


### 실행
- **run_manual_episode** : 한 회차에 대한 에이전트를 통한 설정 충돌 검사 진행.
- **run_multiple_episodes** : 테스트 용이성을 위해 다회차를 한 번에 검사하기 위한 함수. 루프를 돌며 run_manual_episode 호출.
- __main__ : 텍스트 청킹 후 run_multiple_episodes 호출.



