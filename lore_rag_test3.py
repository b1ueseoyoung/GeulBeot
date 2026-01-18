import os
import json
from typing import List, Dict
from collections import Counter
import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==============================
# 0. 기본 설정
# ==============================

os.environ['OPENAI_API_KEY'] = "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" 

STORY_FILE = "story.txt"                    # 전체 원고 파일
VECTOR_DIR = "./chroma_story_chunks"        # ChromaDB 디렉토리
COLLECTION_NAME = "story_chunks"            # 컬렉션 이름
META_JSON_PATH = "story_chunks_meta.json"   # 청크/라벨 저장용 JSON

# 사용할 LLM 모델 (토큰 부담 줄이려고 4o-mini 권장)
CLASSIFY_MODEL = "gpt-4o-mini"


# ==============================
# 1. story.txt 읽기 + 청킹
# ==============================

def load_story(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"story.txt를 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_story(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    print(f"[청킹] story.txt에서 {len(chunks)}개의 청크가 생성되었습니다.")
    return chunks


# ==============================
# 2. A/B/C/D 분류용 LLM
# ==============================

def create_classify_llm() -> ChatOpenAI:
    # 온도 0: 최대한 일관된 분류
    return ChatOpenAI(model=CLASSIFY_MODEL, temperature=0)


def classify_chunk(llm: ChatOpenAI, chunk: str) -> str:
    """
    A: 사실/설정 (세계관 규칙, 인물/아이템의 상태, 사건, 배경 설정 등)
    B: 감정/내면 (인물의 감정, 심리, 내면 독백 등)
    C: 대화 (직접 화법 대사)
    D: 단순 서술 (풍경/분위기 묘사, 설정과 직접적 관련 X)
    """
    system_msg = (
        "너는 웹소설 설정을 분석하는 분류기야.\n"
        "아래 문장을 보고 다음 네 가지 중 하나로 분류해.\n"
        "A: 사실/설정 (세계관 규칙, 인물/아이템의 상태, 사건, 배경 설정 등)\n"
        "B: 감정/내면 (인물의 감정, 심리, 내면 독백 등)\n"
        "C: 대화 (직접 화법 대사)\n"
        "D: 단순 서술 (설정과 크게 상관없는 묘사, 분위기 설명 등)\n"
        "반드시 'A', 'B', 'C', 'D' 중 **한 글자만** 출력해."
    )

    resp = llm.invoke(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"다음 문장을 분류해줘:\n\n{chunk}"},
        ]
    )

    text = resp.content.strip().upper()
    # 혹시 모델이 말이 많아도 안전하게 처리
    for ch in ["A", "B", "C", "D"]:
        if ch in text:
            return ch
    return "D"  # 안전 fallback


def classify_all_chunks(chunks: List[str]) -> List[str]:
    llm = create_classify_llm()
    labels: List[str] = []

    print("\n[A/B/C/D 분류 시작]")
    for i, chunk in enumerate(chunks):
        print(f"  - [{i+1}/{len(chunks)}] 분류 중...", end="", flush=True)
        label = classify_chunk(llm, chunk)
        labels.append(label)
        print(f" → {label}")
        # 너무 빠른 호출로 rate limit 걸리는 것 방지용 (필요시 조정)
        time.sleep(0.05)

    counter = Counter(labels)
    print("\n[분류 결과 요약]")
    for t in ["A", "B", "C", "D"]:
        print(f"  타입 {t}: {counter.get(t, 0)}개")
    return labels


# ==============================
# 3. ChromaDB에 벡터화 + 저장
# ==============================

def build_chroma_with_labels(chunks: List[str], labels: List[str]) -> None:
    if len(chunks) != len(labels):
        raise ValueError("chunks 개수와 labels 개수가 다릅니다.")

    embeddings = OpenAIEmbeddings()

    # 기존 디렉토리가 있으면 지울지 말지 결정 (여기서는 덮어쓰기 위해 삭제 권장)
    if os.path.exists(VECTOR_DIR):
        print(f"[주의] 기존 벡터 디렉토리 {VECTOR_DIR} 가 존재합니다. 새로 덮어씁니다.")
        # 안전하게 테스트 후 필요하면 주석 해제
        import shutil
        shutil.rmtree(VECTOR_DIR)

    print("\n[ChromaDB 생성 및 벡터화 시작]")
    metadatas: List[Dict] = []
    for i, label in enumerate(labels):
        metadatas.append(
            {
                "chunk_index": i,
                "chunk_type": label,  # A/B/C/D
            }
        )

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=VECTOR_DIR,
        collection_name=COLLECTION_NAME,
    )

    # 최신 Chroma는 자동 persist라서 이 라인은 사실 필요 없지만, 명시적으로 한 번 더 호출
    vectordb.persist()
    print("[ChromaDB] 벡터화 및 저장 완료")
    print(f"  - 디렉토리: {VECTOR_DIR}")
    print(f"  - 컬렉션명: {COLLECTION_NAME}")
    print(f"  - 총 청크 수: {len(chunks)}")


# ==============================
# 4. 메타 정보 JSON으로 저장
# ==============================

def save_meta_json(chunks: List[str], labels: List[str]) -> None:
    meta_list = []
    for i, (text, label) in enumerate(zip(chunks, labels)):
        meta_list.append(
            {
                "chunk_index": i,
                "chunk_type": label,
                "text": text,
            }
        )

    with open(META_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

    print(f"\n[JSON 저장] {META_JSON_PATH} 에 청크/타입 정보 저장 완료")


# ==============================
# 5. main
# ==============================

def main():
    print("🤖 Lore Keeper 1단계: story.txt → 청킹 → A/B/C/D 분류 → ChromaDB 적재\n")

    # 1) story.txt 로드
    story_text = load_story(STORY_FILE)

    # 2) 청킹
    chunks = chunk_story(story_text)

    # 3) A/B/C/D 분류
    labels = classify_all_chunks(chunks)

    # 4) ChromaDB에 벡터화 + 저장
    build_chroma_with_labels(chunks, labels)

    # 5) 메타 JSON 저장
    save_meta_json(chunks, labels)

    print("\n✅ 전체 파이프라인 완료!")
    print("   → 이후 RAG + 에이전틱 충돌 검사는 이 ChromaDB와 JSON을 기반으로 만들면 돼.")


if __name__ == "__main__":
    main()
