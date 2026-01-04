import pandas as pd

def mark_conflicts_pandas(
    chunk_file_path: str,
    conflict_db_path: str,
    output_path: str,
):
    # 1. 청킹된 파일 로드 (records 리스트형 JSON)
    chunks = pd.read_json(chunk_file_path)  # 예: gold_conflicts.json

    # 2. conflict_db.jsonl 로드 (jsonl → lines=True)
    conflicts = pd.read_json(conflict_db_path, lines=True)

    # 3. 키로 사용할 컬럼 타입 맞추기 (int로 통일)
    for col in ["source_seq", "chunk_index"]:
        if col in chunks.columns:
            chunks[col] = chunks[col].astype(int)
        if col in conflicts.columns:
            conflicts[col] = conflicts[col].astype(int)

    # 4. (source_seq, chunk_index)만 뽑아서 중복 제거 + is_conflict=True 부여
    conflict_pairs = (
        conflicts[["source_seq", "chunk_index"]]
        .drop_duplicates()
        .copy()
    )
    conflict_pairs["is_conflict"] = True

    # 5. 기존 is_conflict 컬럼은 지우고(있으면), left join으로 매칭
    chunks = chunks.drop(columns=["is_conflict"], errors="ignore")

    merged = chunks.merge(
        conflict_pairs,
        on=["source_seq", "chunk_index"],
        how="left",
    )

    # 매칭된 곳은 True, 나머지는 False
    merged["is_conflict"] = merged["is_conflict"].fillna(False)

    # 6. 저장
    merged.to_json(
        output_path,
        orient="records",
        force_ascii=False,
        indent=2,
    )

    # 간단 로그
    total = len(merged)
    true_cnt = int(merged["is_conflict"].sum())
    print(f"[INFO] 총 {total}개 중 is_conflict=True {true_cnt}개")
    print(f"[DONE] 결과 저장: {output_path}")


if __name__ == "__main__":
    CHUNK_FILE = "base.json"      # 청킹 결과
    CONFLICT_DB = "conflict_db.jsonl"       # 실제 충돌 로그
    OUTPUT_FILE = "conflict_db.json"

    mark_conflicts_pandas(
        chunk_file_path=CHUNK_FILE,
        conflict_db_path=CONFLICT_DB,
        output_path=OUTPUT_FILE,
    )
