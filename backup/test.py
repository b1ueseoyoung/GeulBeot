import os
import json
import pandas as pd
from openai import OpenAI

# ==========================================
# 1. 설정 (API 키와 파일명을 확인하세요)
# ==========================================
client = OpenAI()

SOURCE_FILE = "story.txt" # 혹은 sim-sangnoksu.txt
OUTPUT_JSON = "ground_truth_100_v3.json"
OUTPUT_CSV = "ground_truth_100_v3.csv"
TARGET_COUNT = 100 

# ==========================================
# 2. 소설 텍스트 로딩
# ==========================================
def load_and_split_text(filepath, chunk_size=2000):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        print(f"📖 소설 로딩 완료! 총 {len(chunks)}개의 덩어리로 나눴습니다.")
        return chunks
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {filepath}")
        return []

# ==========================================
# 3. GPT 문제 출제 (프롬프트 대폭 수정됨)
# ==========================================
def generate_questions(chunk_text, num_questions=5):
    prompt = f"""
    너는 웹소설 설정 오류 탐지 시스템을 평가하기 위한 '고난이도 데이터 생성기'야.
    아래 [소설 본문]을 읽고, 설정 충돌 여부를 테스트할 수 있는 문장 {num_questions}개를 만들어줘.
    
    [핵심 규칙 - 매우 중요!]
    1. **비율:** '충돌(True)'과 '정상(False)'을 절반씩 섞어.
    2. **난이도 상향 (Subtle Errors):**
       - '죽은', '사망한', '살아난', '귀신이 된' 같은 **직설적인 표현을 절대 쓰지 마.**
       - 대신, **자연스러운 행동**으로 묘사해. (예: 죽은 사람이 '국밥을 먹었다'거나 '말을 걸었다'고 서술)
       - 독자가 내용을 모르면 정상적인 문장처럼 보여야 해.
    
    3. **충돌 유형 구분 (Conflict Type):**
       - **Hard Conflict:** 시공간, 생사 여부, 명백한 팩트 오류. (예: 부산에 있는 사람이 갑자기 서울에 등장)
       - **Soft Conflict:** 캐릭터의 성격(Persona), 말투, 관계성, 감정선의 오류. (예: 헌신적인 영신이 갑자기 아이들을 귀찮아함)

    [소설 본문 일부]
    {chunk_text}

    [출력 포맷 (JSON)]
    반드시 아래 JSON 포맷만 출력해. root key는 "dataset"이어야 해.
    {{
        "dataset": [
            {{
                "input_text": "영신은 아이들이 시끄럽게굴자 짜증을 내며 회초리를 들었다.",
                "is_conflict": true,
                "conflict_type": "Soft Conflict",
                "reason": "영신은 아이들을 천사처럼 아끼는 헌신적인 성격임. 아이들에게 짜증을 내는 것은 캐릭터 붕괴임.",
                "evidence": "본문에서 영신이 아픈 몸을 이끌고 아이들을 가르치는 장면"
            }},
            {{
                "input_text": "동혁은 회관 건립을 위해 직접 흙짐을 졌다.",
                "is_conflict": false,
                "conflict_type": "None",
                "reason": "동혁은 직접 노동을 하며 솔선수범하는 리더임.",
                "evidence": "회관 낙성 기념으로 나무를 심고 노동하는 장면"
            }}
        ]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "You are a tricky dataset generator. Never use spoiler words like 'dead' in input_text."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.8 # 창의성을 위해 약간 높임
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        if "dataset" in data and isinstance(data["dataset"], list):
            return data["dataset"]
        else:
            return []
        
    except Exception as e:
        print(f"❌ GPT 호출 중 에러: {e}")
        return []

# ==========================================
# 4. 데이터 검증
# ==========================================
def validate_item(item):
    required_keys = ["input_text", "is_conflict", "reason"]
    if not isinstance(item, dict):
        return False
    for key in required_keys:
        if key not in item:
            return False
        value = item[key]
        if isinstance(value, bool):
            continue
        if not value:
            return False
    return True

# ==========================================
# 5. 메인 실행
# ==========================================
def main():
    chunks = load_and_split_text(SOURCE_FILE)
    if not chunks:
        return

    valid_dataset = []
    print("🚀 [Ver 3.0] 고난이도 데이터 생성 시작...")
    
    for i, chunk in enumerate(chunks):
        if len(valid_dataset) >= TARGET_COUNT:
            break
            
        print(f"⏳ [{i+1}/{len(chunks)}] 생성 중... (현재: {len(valid_dataset)}개)")
        generated_items = generate_questions(chunk, num_questions=5)
        
        for item in generated_items:
            if validate_item(item):
                valid_dataset.append(item)

    if valid_dataset:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(valid_dataset, f, indent=4, ensure_ascii=False)
        
        df = pd.DataFrame(valid_dataset)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        
        print("="*50)
        print(f"🎉 생성 완료! 총 {len(valid_dataset)}개")
        print(f"📄 저장 파일: {OUTPUT_CSV}")
        print("="*50)
    else:
        print("😭 데이터 생성 실패")

if __name__ == "__main__":
    main()