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

### DB
| 이름 | 설명 | 형식 |
| --- | --- | --- |
| Lore_DB | 확정된 설정 데이터 | jsonl 파일 + 벡터 DB(FAISS) |
| Full_Story_DB | 소설 원문 청크 데이터 | jsonl 파일 + 벡터 DB(FAISS) |
| Conflict_DB | 충돌 로그 | jsonl 파일 |
| Current_DB | 현재 처리 중인 회차의 설정 데이터 | 단순 List + 임시 벡터 DB(FAISS) |
<br>

#### 관련 함수
- _normalize_enum : 입력값이 정해진 Enum 범위 내에 있는지 확인 후 대문자로 통일. 만약 범위 밖일 경우 기본값 할당.
- _build_lore_item : LLM을 통해 추출한 Fact를 Lore_DB 스키마에 맞춰 변환.
- _load_dataframe / _write_dataframe : jsonl 형식의 파일을 pandas로 dataframe 형식으로 가져오기/저장하기. 쉬운 파일 입출력 관리 목적.
- 

**벡터 스토어**
- get_lore_db_vectordb / get_full_story_db_vectordb : 저장된 Lore_DB, Full_Story_Db 벡터 스토어를 로드. 파일이 없으면 None 반환.
- add_to_lore_db_vectorstore / add_to_full_story_db_vectorstore : 추출된 설정과 원문 청크를 각각 벡터화하여 저장
- save_lore_db_to_file / save_full_story_to_file


- get_current_chunk_vectordb / add_to_current_chunk_vectorstore : 현재 읽고 있는 문장(충돌 검사를 진행할 청크)를 즉시 벡터화. 한 회차 내에서 설정 충돌 검사를 위해 사용.
- get_current_story_vectordb / add_to_current_db_vectorstore : 현재 회차에서 확정된 설정(Lore_Items)을 임시 벡터에 저장. 한 회차 내에서 설정 충돌 검사를 위해 사용.
- reset_current_episode_state : 현재 회차의 모든 임시 데이터를 리셋.






