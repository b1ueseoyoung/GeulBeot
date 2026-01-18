import json
from typing import Dict, Tuple, Any, List

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

Key = Tuple[int, int]  # (source_seq, chunk_index)


def load_json_as_dict(path: str) -> Dict[Key, Dict[str, Any]]:
    """
    conflict_db.json / gold_conflicts.json 공통 로더
    -> (source_seq, chunk_index)를 Key로 하는 dict로 변환
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[Key, Dict[str, Any]] = {}
    for entry in data:
        key = (entry["source_seq"], entry["chunk_index"])
        result[key] = {
            "is_conflict": bool(entry.get("is_conflict", False)),
            "conflict_type": entry.get("conflict_type", "") or "",
            # "text": (entry.get("text") or "").strip(),
        }
    return result


def evaluate(pred: Dict[Key, Dict[str, Any]], gold: Dict[Key, Dict[str, Any]]) -> Dict[str, Any]:
    """
    - conflict 여부: sklearn.metrics 로 accuracy / precision / recall / f1, confusion matrix
    - conflict_type: classification_report / confusion_matrix
    - text(Evidence) span 정확 매칭 비율: 직접 계산
    """
    keys = sorted(set(gold.keys()) | set(pred.keys()))

    # 1) conflict 여부(binary) 평가용 라벨 리스트
    y_true_conflict: List[int] = []
    y_pred_conflict: List[int] = []

    # 2) conflict_type 평가용 라벨 리스트
    y_true_type: List[str] = []
    y_pred_type: List[str] = []

    # 3) span 매칭 계산용
    span_match_tp = 0
    span_match_total = 0

    # 4) 상세 결과 저장용
    details = []

    for key in keys:
        g = gold.get(key, {"is_conflict": False, "conflict_type": ""}) #, "text": ""
        p = pred.get(key, {"is_conflict": False, "conflict_type": ""}) #, "text": ""

        g_c = bool(g["is_conflict"])
        p_c = bool(p["is_conflict"])

        g_type = g.get("conflict_type", "") or ""
        p_type = p.get("conflict_type", "") or ""
        # g_text = g.get("text", "") or ""
        # p_text = p.get("text", "") or ""

        # ----- 1) conflict 여부 -----
        y_true_conflict.append(1 if g_c else 0)
        y_pred_conflict.append(1 if p_c else 0)

        # ----- 2) conflict_type -----
        # gold 또는 pred 중 하나라도 conflict라고 본 건에 대해서만 타입 비교
        if g_c or p_c:
            # 둘 다 non-conflict인 케이스를 "NONE" 클래스처럼 볼 수도 있음
            y_true_type.append(g_type if g_c else "NONE")
            y_pred_type.append(p_type if p_c else "NONE")

        # ----- 3) span 매칭 (TP에 대해서만) -----
        # if g_c and p_c:
        #     span_match_total += 1
        #     if g_text and (g_text == p_text):
        #         span_match_tp += 1

        # ----- 4) 개별 상세 -----
        details.append(
            {
                "source_seq": key[0],
                "chunk_index": key[1],
                "gold_is_conflict": g_c,
                "pred_is_conflict": p_c,
                "gold_type": g_type,
                "pred_type": p_type,
                # "gold_text": g_text,
                # "pred_text": p_text,
            }
        )

    # ====== sklearn.metrics 사용해서 기본 지표 계산 ======
    # Binary conflict 여부
    acc = accuracy_score(y_true_conflict, y_pred_conflict)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_conflict, y_pred_conflict, average="binary", zero_division=0
    )

    # Conflict 여부 classification_report & confusion_matrix
    conflict_report = classification_report(
        y_true_conflict,
        y_pred_conflict,
        target_names=["no_conflict", "conflict"],
        digits=3,
        zero_division=0,
    )
    conflict_cm = confusion_matrix(y_true_conflict, y_pred_conflict)

    # Conflict type classification_report & confusion_matrix
    if y_true_type:
        type_labels = sorted(set(y_true_type + y_pred_type))
        type_report = classification_report(
            y_true_type,
            y_pred_type,
            labels=type_labels,
            digits=3,
            zero_division=0,
        )
        type_cm = confusion_matrix(y_true_type, y_pred_type, labels=type_labels)
    else:
        type_labels = []
        type_report = ""
        type_cm = []

    # Span 매칭 비율
    span_match_rate = span_match_tp / span_match_total if span_match_total > 0 else 0.0

    return {
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "span_match_rate": span_match_rate,
        },
        "conflict_report": conflict_report,
        "conflict_confusion_matrix": conflict_cm.tolist(),
        "type_labels": type_labels,
        "type_report": type_report,
        "type_confusion_matrix": type_cm.tolist() if len(type_labels) > 0 else [],
        "details": details,
    }


if __name__ == "__main__":
    # 1) 예측 / 정답 로드
    pred = load_json_as_dict("conflict_db.json")       # 모델이 낸 결과
    gold = load_json_as_dict("gold_conflicts.json")    # 사람이 만든 정답셋

    # 2) 평가 수행
    result = evaluate(pred, gold)

    # 3) 출력
    print("=== Basic Metrics (conflict 여부) ===")
    print(f"accuracy : {result['metrics']['accuracy']:.3f}")
    print(f"precision: {result['metrics']['precision']:.3f}")
    print(f"recall   : {result['metrics']['recall']:.3f}")
    print(f"f1       : {result['metrics']['f1']:.3f}")
    print(f"span-level exact match (text): {result['metrics']['span_match_rate']:.3f}")
    print()

    print("=== Conflict Detection Report (binary) ===")
    print(result["conflict_report"])
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(result["conflict_confusion_matrix"])
    print()

    print("=== Conflict Type Classification Report ===")
    if result["type_labels"]:
        print("Labels:", result["type_labels"])
        print(result["type_report"])
        print("Confusion matrix (rows=gold, cols=pred, same order as labels):")
        print(result["type_confusion_matrix"])
    else:
        print("No conflict cases to evaluate type classification.")
    print()

    # 4) 개별 청크별 상세 결과 저장 (원래 코드와 비슷하게)
    with open("eval_details.json", "w", encoding="utf-8") as f:
        json.dump(result["details"], f, ensure_ascii=False, indent=2)
    print("[eval_details.json]으로 개별 청크 결과 저장 완료")
