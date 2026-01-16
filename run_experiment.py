"""
==============================================================================
CO-DITOR 실험 자동화 스크립트 (run_experiment.py)
==============================================================================

목적:
    FFF 데이터셋(flawed_fictions_long)의 각 샘플에 대해 CO-DITOR를 실행하고,
    결과를 FFF 평가 포맷으로 변환하여 Baseline과 비교할 수 있게 합니다.

실행 방법:
    python run_experiment.py --num_samples 10  # 처음 10개 샘플만 테스트
    python run_experiment.py                    # 전체 200개 샘플 실행

출력:
    - experiment_results/coditor_results.json   : CO-DITOR 결과 (FFF 포맷)
    - experiment_results/metrics_summary.json   : 성능 지표 요약
    - experiment_results/detailed_logs/         : 샘플별 상세 로그

==============================================================================
"""

import os
import sys
import json
import shutil
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 데이터셋 및 NLP 도구
from datasets import load_dataset, Dataset
import pandas as pd
import nltk

# FFF 공식 evaluator
from evaluate.localization_metrics import eval_cont_error_localization

# NLTK 데이터 다운로드 (최초 1회)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# newversion 모듈 임포트 (전역 - reset_newversion_globals에서 사용)
import newversion as nv

# ==============================================================================
# 설정 상수
# ==============================================================================

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiment_results")
DETAILED_LOGS_DIR = os.path.join(RESULTS_DIR, "detailed_logs")

# CSV 데이터셋 경로
CSV_DATASET_PATH = os.path.join(PROJECT_ROOT, "flawed_fictions_DataSet.csv")

# DB 파일/폴더 경로 (초기화 대상)
DB_PATHS = {
    "faiss_lore_db": os.path.join(PROJECT_ROOT, "faiss_lore_db"),
    "faiss_full_story_db": os.path.join(PROJECT_ROOT, "faiss_full_story_db"),
    "lore_db.jsonl": os.path.join(PROJECT_ROOT, "lore_db.jsonl"),
    "full_story_db.jsonl": os.path.join(PROJECT_ROOT, "full_story_db.jsonl"),
    "conflict_db.jsonl": os.path.join(PROJECT_ROOT, "conflict_db.jsonl"),
}

# 청크 분할 설정 (Spec 권장값)
CHUNK_SIZE = 600        # 500~800 권장
CHUNK_OVERLAP = 80      # 50~100 권장


# ==============================================================================
# 1단계: DB 초기화 함수
# ==============================================================================

def reset_all_databases() -> bool:
    """
    모든 DB를 완전히 초기화합니다.
    
    초기화 대상:
        - faiss_lore_db/ (디렉토리)
        - faiss_full_story_db/ (디렉토리)
        - lore_db.jsonl (파일)
        - full_story_db.jsonl (파일)
        - conflict_db.jsonl (파일)
    
    Returns:
        bool: 초기화 성공 여부
    """
    print("\n[DB 초기화] 시작...")
    
    success = True
    for name, path in DB_PATHS.items():
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  ✓ 디렉토리 삭제: {name}")
            elif os.path.isfile(path):
                os.remove(path)
                print(f"  ✓ 파일 삭제: {name}")
            else:
                print(f"  - 존재하지 않음 (스킵): {name}")
        except Exception as e:
            print(f"  ✗ 삭제 실패: {name} - {e}")
            success = False
    
    # newversion.py의 전역 변수도 초기화 (모듈 리로드)
    # 주의: 이 부분은 newversion.py 임포트 후에 호출해야 함
    
    print(f"[DB 초기화] {'완료 ✓' if success else '일부 실패 ✗'}\n")
    return success


def reset_newversion_globals():
    """
    newversion.py의 전역 상태 변수를 초기화합니다.
    
    매 샘플 처리 전에 호출하여 이전 샘플의 데이터가 남지 않도록 합니다.
    """
    # 리스트 초기화
    nv.current_story_db.clear()
    nv.conflict_db.clear()
    nv.lore_db.clear()
    nv.full_story_db.clear()
    
    # 벡터스토어 초기화
    nv._lore_db_vectordb = None
    nv._full_story_db_vectordb = None
    nv._current_story_vectordb = None
    nv._current_chunk_vectordb = None
    
    # 상태 플래그 초기화
    nv._state_loaded = False
    
    # 청크 컨텍스트 초기화
    nv.SEARCH_CONTEXT_BY_CHUNK.clear()
    
    print("  ✓ newversion.py 전역 변수 초기화 완료")


# ==============================================================================
# 2단계: HuggingFace 데이터셋 로드 및 CSV 필터링
# ==============================================================================

def load_and_filter_fff_dataset(csv_path: str) -> Tuple[Dataset, List[str]]:
    """
    HuggingFace에서 FFF 데이터셋을 로드하고, CSV 파일의 ID로 필터링합니다.
    
    Args:
        csv_path: 필터링할 ID 목록이 있는 CSV 파일 경로
    
    Returns:
        (filtered_dataset, filtered_ids) 튜플
        - filtered_dataset: 필터링된 HuggingFace Dataset 객체
        - filtered_ids: 필터링된 샘플 ID 리스트
    """
    print("\n[데이터셋 로드]")
    
    # 1) CSV에서 필터링할 ID 추출
    print(f"  - CSV 파일 로드: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8")
    filter_ids = df["id"].tolist()
    print(f"  - 필터링할 샘플 수: {len(filter_ids)}")
    
    # 2) HuggingFace에서 FFF 데이터셋 로드
    print("  - HuggingFace 데이터셋 로드 중...")
    dataset_long = load_dataset("kahuja/flawed-fictions", split="flawed_fictions_long")
    dataset_cf = load_dataset("kahuja/flawed-fictions", split="flawed_fictions_cf_negs")
    
    print(f"    ✓ flawed_fictions_long: {len(dataset_long)} 샘플")
    print(f"    ✓ flawed_fictions_cf_negs: {len(dataset_cf)} 샘플")
    
    # 3) 두 데이터셋 합치기
    from datasets import concatenate_datasets
    combined_dataset = concatenate_datasets([dataset_long, dataset_cf])
    print(f"  - 전체 데이터셋: {len(combined_dataset)} 샘플")
    
    # 4) CSV의 ID로 필터링
    # HuggingFace 데이터셋의 'example_id' 필드와 CSV의 'id' 매칭
    filtered_dataset = combined_dataset.filter(
        lambda example: example["example_id"] in filter_ids
    )
    
    print(f"  - 필터링 완료: {len(filtered_dataset)} 샘플")
    
    # 5) 필터링된 ID 순서 (CSV 순서 유지)
    filtered_ids = [ex["example_id"] for ex in filtered_dataset]
    
    return filtered_dataset, filtered_ids


# ==============================================================================
# 3단계: 문장 분할 함수 (FFF 호환)
# ==============================================================================

def split_into_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분할합니다.
    
    FFF 공식 평가와의 호환성을 위해 NLTK sent_tokenize 사용.
    
    Args:
        text: 분할할 텍스트
    
    Returns:
        문장 리스트 (각 문장에 ID 부여 가능)
    
    Example:
        >>> sentences = split_into_sentences("Hello world. How are you?")
        >>> print(sentences)
        ['Hello world.', 'How are you?']
    """
    # NLTK sent_tokenize 사용 (영어 기준)
    sentences = sent_tokenize(text)
    return sentences


def create_sentence_id_mapping(sentences: List[str]) -> Dict[int, str]:
    """
    문장 ID → 문장 텍스트 매핑을 생성합니다.
    
    Args:
        sentences: 문장 리스트
    
    Returns:
        {sentence_id: sentence_text} 딕셔너리
    """
    return {i: sent for i, sent in enumerate(sentences)}


# ==============================================================================
# 4단계: 청크 분할 함수
# ==============================================================================

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    텍스트를 청크 단위로 분할합니다.
    
    Args:
        text: 분할할 텍스트
        chunk_size: 청크 최대 크기 (기본 600자)
        chunk_overlap: 청크 간 겹침 (기본 80자)
    
    Returns:
        청크 리스트
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    
    chunks = splitter.split_text(text)
    return chunks


def build_chunk_to_sentence_mapping(
    chunks: List[str],
    sentences: List[str]
) -> Dict[int, List[int]]:
    """
    청크 인덱스 → 문장 ID 리스트 매핑을 생성합니다.
    
    CO-DITOR가 청크 단위로 충돌을 탐지하므로,
    충돌이 발생한 청크가 어떤 문장들을 포함하는지 매핑이 필요합니다.
    
    Args:
        chunks: 청크 리스트
        sentences: 문장 리스트
    
    Returns:
        {chunk_index: [sentence_ids]} 딕셔너리
    """
    mapping = {}
    
    for chunk_idx, chunk in enumerate(chunks):
        sentence_ids = []
        for sent_idx, sentence in enumerate(sentences):
            # 문장이 청크에 포함되어 있는지 확인
            # (부분 매칭도 허용 - 청크 경계에서 문장이 잘릴 수 있음)
            if sentence in chunk or chunk in sentence:
                sentence_ids.append(sent_idx)
            elif len(sentence) > 20:  # 짧은 문장 제외
                # 문장의 핵심 부분이 청크에 있는지 확인
                core = sentence[10:-10] if len(sentence) > 30 else sentence
                if core in chunk:
                    sentence_ids.append(sent_idx)
        
        mapping[chunk_idx] = sentence_ids
    
    return mapping


# ==============================================================================
# 5단계: CO-DITOR 실행 함수
# ==============================================================================

def run_coditor_on_story(
    story_text: str,
    story_id: str,
    novel_id: int = 0
) -> Dict[str, Any]:
    """
    단일 스토리에 대해 CO-DITOR를 실행합니다.
    
    Args:
        story_text: 스토리 전체 텍스트
        story_id: 스토리 식별자
        novel_id: 소설 ID (내부 사용)
    
    Returns:
        {
            "story_id": str,
            "sentences": List[str],
            "chunks": List[str],
            "chunk_to_sentence_map": Dict[int, List[int]],
            "conflicts": List[Dict],  # conflict_db 내용
            "summaries": List[Dict],  # 에피소드 요약
        }
    """
    print(f"\n{'='*60}")
    print(f"[CO-DITOR 실행] Story: {story_id}")
    print(f"{'='*60}")
    
    # 1) 문장 분할
    sentences = split_into_sentences(story_text)
    print(f"  - 문장 수: {len(sentences)}")
    
    # 2) 청크 분할
    chunks = split_into_chunks(story_text)
    print(f"  - 청크 수: {len(chunks)}")
    
    # 3) 청크 → 문장 매핑
    chunk_to_sentence_map = build_chunk_to_sentence_mapping(chunks, sentences)
    
    # 4) CO-DITOR 실행
    # 스토리 전체를 하나의 "에피소드"로 처리
    # (여러 에피소드로 나눌 수도 있지만, 단일 스토리이므로 1개로)
    all_episode_chunks = [chunks]  # 1회차 = 전체 스토리
    
    try:
        summaries, all_results = nv.run_multiple_episodes(
            all_episode_chunks=all_episode_chunks,
            novel_id=novel_id,
        )
    except Exception as e:
        print(f"  ✗ CO-DITOR 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "story_id": story_id,
            "error": str(e),
            "sentences": sentences,
            "chunks": chunks,
            "chunk_to_sentence_map": chunk_to_sentence_map,
            "conflicts": [],
            "summaries": [],
        }
    
    # 5) 결과 수집
    # conflict_db는 newversion.py의 전역 변수
    conflicts = list(nv.conflict_db)  # 복사
    
    print(f"  - 탐지된 충돌 수: {len(conflicts)}")
    
    return {
        "story_id": story_id,
        "sentences": sentences,
        "chunks": chunks,
        "chunk_to_sentence_map": chunk_to_sentence_map,
        "conflicts": conflicts,
        "summaries": summaries,
    }


# ==============================================================================
# 6단계: 결과 변환 함수 (FFF 포맷)
# ==============================================================================

def convert_to_fff_format(coditor_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    CO-DITOR 결과를 FFF 평가 포맷으로 변환합니다.
    
    FFF 포맷:
    {
        "story_id": "...",
        "prediction": 0 or 1,           # 충돌 있음(1) / 없음(0)
        "error_sentence_ids": [...],    # 오류가 있는 문장 ID들
        "contr_sentence_ids": [...],    # 충돌 증거 문장 ID들
        "reasoning": "...",             # 판단 근거
        "evidence": "..."               # 증거 텍스트
    }
    
    Args:
        coditor_result: run_coditor_on_story()의 반환값
    
    Returns:
        FFF 평가 포맷 딕셔너리
    """
    story_id = coditor_result["story_id"]
    conflicts = coditor_result["conflicts"]
    sentences = coditor_result["sentences"]
    chunk_to_sentence_map = coditor_result["chunk_to_sentence_map"]
    
    # 충돌 여부
    has_conflict = len(conflicts) > 0
    prediction = 1 if has_conflict else 0
    
    # 문장 ID 추출
    error_sentence_ids = []
    contr_sentence_ids = []
    reasoning_parts = []
    evidence_parts = []
    
    for conflict in conflicts:
        # 1) 충돌 발생 청크 → 문장 ID
        chunk_idx = conflict.get("chunk_index", -1)
        if chunk_idx in chunk_to_sentence_map:
            error_sentence_ids.extend(chunk_to_sentence_map[chunk_idx])
        
        # 2) 충돌 증거 텍스트 → 문장 ID
        conflicting_text = conflict.get("conflicting_text", "")
        if conflicting_text:
            for sent_idx, sent in enumerate(sentences):
                # 증거 텍스트가 문장에 포함되어 있으면
                if conflicting_text in sent or sent in conflicting_text:
                    contr_sentence_ids.append(sent_idx)
                # 부분 매칭 (긴 텍스트의 경우)
                elif len(conflicting_text) > 30:
                    core = conflicting_text[:50]
                    if core in sent:
                        contr_sentence_ids.append(sent_idx)
        
        # 3) 추론 근거 수집
        evidence = conflict.get("evidence", "")
        if evidence:
            reasoning_parts.append(evidence)
        
        # 4) 증거 텍스트 수집
        if conflicting_text:
            evidence_parts.append(conflicting_text)
    
    # 중복 제거 및 정렬
    error_sentence_ids = sorted(set(error_sentence_ids))
    contr_sentence_ids = sorted(set(contr_sentence_ids))
    
    # 문장 ID를 문자열 리스트로 변환 (FFF 포맷 호환)
    # FFF에서는 "1,2,3" 형식일 수 있음 - 확인 필요
    error_lines_str = ",".join(map(str, error_sentence_ids)) if error_sentence_ids else "NA"
    contr_lines_str = ",".join(map(str, contr_sentence_ids)) if contr_sentence_ids else "NA"
    
    return {
        "story_id": story_id,
        "prediction": prediction,
        "error_sentence_ids": error_sentence_ids,
        "contr_sentence_ids": contr_sentence_ids,
        "error_lines": error_lines_str,      # FFF 원본 포맷 호환
        "contradicted_lines": contr_lines_str,  # FFF 원본 포맷 호환
        "reasoning": " | ".join(filter(None, reasoning_parts))[:500],  # 500자 제한
        "evidence": " | ".join(filter(None, evidence_parts))[:500],
    }


# ==============================================================================
# 7단계: 평가 지표 계산 함수 (FFF 공식 evaluator 사용)
# ==============================================================================

def convert_to_fff_evaluator_format(
    predictions: List[Dict[str, Any]],
    sentences_list: List[List[str]]
) -> List[Dict[str, Any]]:
    """
    CO-DITOR 결과를 FFF 공식 evaluator 입력 형식으로 변환합니다.
    
    Args:
        predictions: CO-DITOR 예측 결과 (story_id, prediction, error_sentence_ids, contr_sentence_ids)
        sentences_list: 각 스토리의 문장 리스트
    
    Returns:
        FFF evaluator 형식의 결과 리스트
        [
            {
                "cont_error": float,
                "cont_error_expl": str,
                "cont_error_lines": List[str],
                "contradicted_lines": List[str],
            },
            ...
        ]
    """
    fff_results = []
    
    # predictions와 sentences_list는 같은 순서로 저장되어 있으므로 인덱스로 매칭
    for idx, pred in enumerate(predictions):
        # 해당 스토리의 문장 리스트 가져오기
        if idx >= len(sentences_list):
            print(f"  [경고] 인덱스 {idx}의 문장 리스트 없음")
            continue
        
        sentences = sentences_list[idx]
        
        # 문장 ID를 문장 텍스트로 변환
        error_lines = []
        for sent_id in pred.get("error_sentence_ids", []):
            if 0 <= sent_id < len(sentences):
                error_lines.append(sentences[sent_id])
        
        contradicted_lines = []
        for sent_id in pred.get("contr_sentence_ids", []):
            if 0 <= sent_id < len(sentences):
                contradicted_lines.append(sentences[sent_id])
        
        fff_results.append({
            "cont_error": float(pred["prediction"]),
            "cont_error_expl": pred.get("reasoning", ""),
            "cont_error_lines": error_lines,
            "contradicted_lines": contradicted_lines,
        })
    
    return fff_results


def calculate_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    sentences_list: List[List[str]],
    dataset: Any,
    start_idx: int = 0
) -> Dict[str, float]:
    """
    FFF 공식 evaluator를 사용하여 평가 지표를 계산합니다.
    
    Args:
        predictions: CO-DITOR 예측 결과 리스트
        ground_truth: FFF 데이터셋의 정답 리스트
        sentences_list: 각 스토리의 문장 리스트
        dataset: HuggingFace dataset 객체 (전체)
        start_idx: 시작 인덱스
    
    Returns:
        {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float,
            "ceeval_full": float,
            "ceeval_pos": float,
        }
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from datasets import Dataset
    
    # 1) Classification 지표 계산 (sklearn 사용)
    y_true = []
    y_pred = []
    
    # story_id로 매칭
    gt_dict = {gt["story_id"]: gt for gt in ground_truth}
    
    for pred in predictions:
        story_id = pred["story_id"]
        if story_id not in gt_dict:
            print(f"  [경고] Ground truth 없음: {story_id}")
            continue
        
        gt = gt_dict[story_id]
        
        # 정답: cont_error 필드 (0.0 또는 1.0)
        y_true.append(int(gt.get("cont_error", 0)))
        y_pred.append(pred["prediction"])
    
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ceeval_full": 0.0,
            "ceeval_pos": 0.0,
        }
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    # 2) CEEval 지표 계산 (FFF 공식 evaluator 사용)
    try:
        # CO-DITOR 결과를 FFF evaluator 형식으로 변환
        fff_format_results = convert_to_fff_evaluator_format(predictions, sentences_list)
        
        # 실제 처리한 샘플에 해당하는 dataset 부분만 추출
        # FFF evaluator는 인덱스 기반으로 매칭하므로, 
        # 처리한 샘플 범위의 dataset을 새로 만듦
        num_samples = len(predictions)
        dataset_subset_dict = []
        for i in range(num_samples):
            # dataset은 이미 CSV로 필터링된 상태이므로 start_idx를 그대로 사용
            dataset_subset_dict.append(dataset[start_idx + i])
        
        # Dataset 객체로 변환
        dataset_subset = Dataset.from_list(dataset_subset_dict)
        
        # FFF 공식 evaluator 호출
        _, ceeval_full, _, _ = eval_cont_error_localization(
            fff_format_results, dataset_subset, pos_only=False
        )
        
        _, ceeval_pos, _, _ = eval_cont_error_localization(
            fff_format_results, dataset_subset, pos_only=True
        )
        
    except Exception as e:
        print(f"  [경고] CEEval 계산 실패: {e}")
        import traceback
        traceback.print_exc()
        ceeval_full = 0.0
        ceeval_pos = 0.0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "ceeval_full": round(float(ceeval_full), 4),
        "ceeval_pos": round(float(ceeval_pos), 4),
    }


# ==============================================================================
# 8단계: 메인 실험 루프
# ==============================================================================

def run_experiment(
    num_samples: Optional[int] = None,
    start_idx: int = 0,
    save_detailed_logs: bool = True
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    전체 실험을 실행합니다.
    
    Args:
        num_samples: 처리할 샘플 수 (None이면 전체)
        start_idx: 시작 인덱스
        save_detailed_logs: 샘플별 상세 로그 저장 여부
    
    Returns:
        (predictions, metrics) 튜플
    """
    print("\n" + "="*70)
    print("CO-DITOR 실험 자동화 시작")
    print("="*70)
    
    # 1) 결과 디렉토리 생성
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if save_detailed_logs:
        os.makedirs(DETAILED_LOGS_DIR, exist_ok=True)
    
    # 2) 데이터셋 로드 (HuggingFace + CSV 필터링)
    print("\n[1단계] HuggingFace 데이터셋 로드 및 CSV 필터링...")
    if not os.path.exists(CSV_DATASET_PATH):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {CSV_DATASET_PATH}")
    
    # HuggingFace 데이터셋 로드 및 CSV ID로 필터링
    dataset, filtered_ids = load_and_filter_fff_dataset(CSV_DATASET_PATH)
    total_samples = len(dataset)
    
    print(f"  - 최종 샘플 수: {total_samples}")
    
    # 처리할 샘플 범위 결정
    end_idx = min(start_idx + num_samples, total_samples) if num_samples else total_samples
    samples_to_process = list(range(start_idx, end_idx))
    print(f"  - 처리할 샘플: {start_idx} ~ {end_idx-1} ({len(samples_to_process)}개)")
    
    # 3) 실험 루프
    predictions = []
    ground_truth = []
    sentences_list = []  # CEEval 계산을 위한 문장 리스트 저장
    
    for i, sample_idx in enumerate(samples_to_process):
        # HuggingFace Dataset에서 샘플 가져오기
        sample = dataset[sample_idx]
        story_id = sample["example_id"]
        story_text = sample["story"]
        
        print(f"\n{'#'*70}")
        print(f"[샘플 {i+1}/{len(samples_to_process)}] {story_id}")
        print(f"{'#'*70}")
        
        # 3-1) DB 초기화
        reset_all_databases()
        reset_newversion_globals()
        
        # 3-2) CO-DITOR 실행
        start_time = time.time()
        coditor_result = run_coditor_on_story(
            story_text=story_text,
            story_id=story_id,
            novel_id=sample_idx,
        )
        elapsed = time.time() - start_time
        print(f"  - 처리 시간: {elapsed:.2f}초")
        
        # 3-3) FFF 포맷 변환
        fff_result = convert_to_fff_format(coditor_result)
        predictions.append(fff_result)
        
        # 3-3.5) 문장 리스트 저장 (CEEval 계산용)
        sentences_list.append(coditor_result["sentences"])
        
        # 3-4) Ground Truth 수집 (HuggingFace Dataset 사용)
        ground_truth.append({
            "story_id": story_id,
            "cont_error": float(sample["cont_error"]),
            "cont_error_expl": sample.get("cont_error_expl", ""),
            "cont_error_lines": sample.get("cont_error_lines", []),
            "contradicted_lines": sample.get("contradicted_lines", []),
        })
        
        # 3-5) 상세 로그 저장
        if save_detailed_logs:
            log_path = os.path.join(DETAILED_LOGS_DIR, f"{story_id}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({
                    "coditor_result": {
                        k: v for k, v in coditor_result.items()
                        if k not in ["sentences", "chunks"]  # 큰 데이터 제외
                    },
                    "fff_result": fff_result,
                    "ground_truth": ground_truth[-1],
                    "elapsed_seconds": elapsed,
                }, f, ensure_ascii=False, indent=2)
        
        # 3-6) 중간 결과 출력
        gt_label = int(sample["cont_error"])
        pred_label = fff_result["prediction"]
        match = "✓" if gt_label == pred_label else "✗"
        print(f"  - 정답: {gt_label}, 예측: {pred_label} {match}")
    
    # 4) 전체 결과 저장
    print("\n[4단계] 결과 저장...")
    
    # 예측 결과 저장
    results_path = os.path.join(RESULTS_DIR, "coditor_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"  - 예측 결과: {results_path}")
    
    # Ground Truth 저장
    gt_path = os.path.join(RESULTS_DIR, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    print(f"  - Ground Truth: {gt_path}")
    
    # 5) 평가 지표 계산
    print("\n[5단계] 평가 지표 계산...")
    print("  - HuggingFace 데이터셋(FFF 형식)으로 평가합니다.")
    print("  - Classification 및 CEEval 지표를 모두 계산합니다.")
    
    # 처리한 샘플 범위에 해당하는 Dataset 추출
    dataset_subset_list = [dataset[idx] for idx in samples_to_process]
    dataset_subset = Dataset.from_list(dataset_subset_list)
    
    # calculate_metrics 함수 호출
    metrics = calculate_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        sentences_list=sentences_list,
        dataset=dataset_subset,
        start_idx=0,  # subset이므로 0부터 시작
    )
    
    # 지표 저장
    metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    metrics_with_meta = {
        "metrics": metrics,
        "experiment_info": {
            "total_samples": len(predictions),
            "timestamp": datetime.now().isoformat(),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_with_meta, f, ensure_ascii=False, indent=2)
    print(f"  - 평가 지표: {metrics_path}")
    
    # 6) 최종 결과 출력
    print("\n" + "="*70)
    print("실험 완료!")
    print("="*70)
    print(f"\n📊 성능 지표 (FFF 공식 evaluator):")
    print(f"  - Accuracy:     {metrics['accuracy']}")
    print(f"  - Precision:    {metrics['precision']}")
    print(f"  - Recall:       {metrics['recall']}")
    print(f"  - F1 Score:     {metrics['f1']}")
    print(f"  - CEEval-Full:  {metrics['ceeval_full']}")
    print(f"  - CEEval-Pos:   {metrics['ceeval_pos']}")
    
    return predictions, metrics


# ==============================================================================
# 실행 진입점
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CO-DITOR 실험 자동화")
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=None,
        help="처리할 샘플 수 (기본값: 전체)"
    )
    parser.add_argument(
        "--start_idx", "-s",
        type=int,
        default=0,
        help="시작 인덱스 (기본값: 0)"
    )
    parser.add_argument(
        "--no_detailed_logs",
        action="store_true",
        help="상세 로그 저장 안 함"
    )
    args = parser.parse_args()

    # 사용 예시 안내
    if args.num_samples is not None:
        print(f"샘플 범위: {args.start_idx} ~ {args.start_idx + args.num_samples - 1} (총 {args.num_samples}개)")
        print("실행 예시: python run_experiment.py --start_idx 5 --num_samples 5")

    # 실험 실행
    predictions, metrics = run_experiment(
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        save_detailed_logs=not args.no_detailed_logs,
    )
