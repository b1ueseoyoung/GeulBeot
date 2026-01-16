import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from nltk import sent_tokenize
import re


def preprocess_sent(sent):
    # Lower case
    sent = sent.lower()

    # Remove weird symbols
    sent = sent.replace("•", "")
    sent = sent.replace(""", "").replace(""", "")

    # Get rid of additional information enclosed in []
    sent = re.sub(r"\[.*?\]", "", sent)
    # Get rid of html tags such as <li>, <br>, <p> etc.
    sent = (
        sent.replace("<li>", " ")
        .replace("</li>", " ")
        .replace("<br>", " ")
        .replace("<br/>", " ")
        .replace("<p>", " ")
        .replace("</p>", " ")
        .replace("<m>", " ")
        .replace("</m>", " ")
    )

    # Remove punctuation
    sent = re.sub(r"[^\w\s]", "", sent)

    # Remove extra whitespace and newlines
    sent = re.sub(r"\s+", " ", sent)
    sent = sent.strip()

    return sent


def extract_full_line(story, predicted_line, get_idxs=False):
    # Remove additional information enclosed in []
    story = re.sub(r"\[.*?\]", "", story)
    predicted_line = re.sub(r"\[.*?\]", "", predicted_line)
    lines = sent_tokenize(
        story.replace("<li>", " ")
        .replace("</li>", " ")
        .replace("<br>", " ")
        .replace("<br/>", " ")
        .replace("<p>", " ")
        .replace("</p>", " ")
        .replace("<m>", " ")
        .replace("</m>", " ")
    )
    predicted_lines = sent_tokenize(predicted_line)
    full_lines = []
    idxs = []
    for idx, line in enumerate(lines):
        for pline in predicted_lines:
            # Use fuzzy matching with a threshold
            processed_line = preprocess_sent(line)
            processed_pline = preprocess_sent(pline)

            # Check exact containment first
            if processed_pline in processed_line:
                full_lines.append(line)
                idxs.append(idx)
    if get_idxs:
        return full_lines, idxs
    return full_lines


def extract_all_full_lines(story, predicted_lines):
    full_lines = []
    for line in predicted_lines:
        full_lines += extract_full_line(story, line)
    return full_lines


def get_cont_error_lines(response):
    """
    Given response from the model with the following format:

    <response>
    <explanation>
    [Provide your explanation here, whether you found a continuity error or not]
    </explanation>

    <error_lines>
    [If applicable, quote the lines that introduce the continuity error]
    </error_lines>

    <contradicted_lines>
    [If applicable, quote the lines from earlier in the story that are contradicted by the error]
    </contradicted_lines>

    <decision>
    [State your final decision on whether a continuity error exists in the story. State "No continuity error found" if you think there is no continuity error.]
    </decision>
    </response>
    
    Extract the lines that introduce the continuity error.
    
    """
    if "<error_lines>" in response:
        error_lines = (
            response.split("<error_lines>")[1]
            .split("</error_lines>")[0]
            .strip()
        )
        error_lines = error_lines.split("\n")
        error_lines = [line.strip().replace("-", "").replace("*", "").strip() for line in error_lines]
        error_lines = [line for line in error_lines if line != ""]
        return error_lines
    else:
        return []


def get_contradicted_lines(response):
    """
    Given response from the model with the following format:

    <response>
    <explanation>
    [Provide your explanation here, whether you found a continuity error or not]
    </explanation>

    <error_lines>
    [If applicable, quote the lines that introduce the continuity error]
    </error_lines>

    <contradicted_lines>
    [If applicable, quote the lines from earlier in the story that are contradicted by the error]
    </contradicted_lines>

    <decision>
    [State your final decision on whether a continuity error exists in the story. State "No continuity error found" if you think there is no continuity error.]
    </decision>
    </response>
    
    Extract the lines that are contradicted by the error.

    """

    if "<contradicted_lines>" in response:
        contradicted_lines = (
            response.split("<contradicted_lines>")[1]
            .split("</contradicted_lines>")[0]
            .strip()
        )
        contradicted_lines = contradicted_lines.split("\n")
        contradicted_lines = [
            line.strip().replace("-", "").replace("*", "").strip()
            for line in contradicted_lines
        ]
        contradicted_lines = [line for line in contradicted_lines if line != ""]
        return contradicted_lines
    else:
        return []

def listify_lines(lines):
    if isinstance(lines, str):
        if "<li>" in lines:
            lines = lines.split("</li>")
            lines = [line.strip().replace("-", "").replace("*", "").strip() for line in lines]
            lines = [line for line in lines if line != ""]
        else:
            lines = lines.split("\n")
            lines = [line.strip().replace("-", "").replace("*", "").strip() for line in lines]
            lines = [line for line in lines if line != ""]
    return lines


def construct_eval_dataset(results, dataset):
    """
    Construct a pandas dataframe with the following columns:
    - story_id
    - story
    - predicted_cont_error
    - ground_truth_cont_error
    - predicted_cont_error_lines
    - predicted_cont_error_lines_full
    - predicted_contradicted_lines
    - predicted_contradicted_lines_full
    - ground_truth_cont_error_lines
    - ground_truth_cont_error_lines_full
    - ground_truth_contradicted_lines
    - predicted_expl
    - ground_truth_expl
    - num_reasoning_tokens
    - num_completion_tokens
    - num_prompt_tokens
    """

    detailed_results = results#["detailed_results"]
    eval_dataset = []

    for idx, result in tqdm(enumerate(detailed_results), total=len(detailed_results), desc="Constructing eval dataset"):

        agg_result = result

        story_id = idx  # agg_result["example_idx"]
        story = dataset[story_id]["story"]

        try:
            num_reasoning_tokens = agg_result["num_reasoning_tokens"]
            num_prompt_tokens = agg_result["num_prompt_tokens"]
            num_completion_tokens = agg_result["num_completion_tokens"]
        except KeyError:
            num_reasoning_tokens = None
            num_prompt_tokens = None
            num_completion_tokens = None
        ground_truth_cont_error = dataset[idx][
            "cont_error"
        ]  # agg_result["cont_error"]["ground_truth"]
        predicted_cont_error = agg_result["cont_error"]
        ground_truth_expl = dataset[idx]["cont_error_expl"]
        predicted_expl = agg_result["cont_error_expl"]

        if ground_truth_cont_error:
            gt_cont_error_lines = listify_lines(dataset[idx]["cont_error_lines"])
            gt_contradicted_lines = listify_lines(dataset[idx]["contradicted_lines"])

            gt_cont_error_lines_full = extract_all_full_lines(
                story, gt_cont_error_lines
            )
            gt_contradicted_lines_full = extract_all_full_lines(
                story, gt_contradicted_lines
            )

            if gt_cont_error_lines_full == []:
                # breakpoint()
                print(
                    f"Failed to extract ground truth cont error lines for story {story_id} :("
                )
                print("-----------")
            if gt_contradicted_lines_full == []:
                # breakpoint()
                print(
                    f"Failed to extract ground truth contradicted lines for story {story_id} :("
                )
                print("-----------")
                print()
        else:
            gt_cont_error_lines = []
            gt_contradicted_lines = []
            gt_cont_error_lines_full = []
            gt_contradicted_lines_full = []

        if predicted_cont_error:
            predicted_cont_error_lines = listify_lines(agg_result["cont_error_lines"])
            predicted_contradicted_lines = listify_lines(
                agg_result["contradicted_lines"]
            )

            predicted_cont_error_lines_full = extract_all_full_lines(
                story, predicted_cont_error_lines
            )
            predicted_contradicted_lines_full = extract_all_full_lines(
                story, predicted_contradicted_lines
            )
            if predicted_cont_error_lines_full == []:
                # breakpoint()
                print(
                    f"Failed to extract predicted cont error lines for story {story_id} :("
                )
                print("-----------")
            if predicted_contradicted_lines_full == []:
                # breakpoint()
                print(
                    f"Failed to extract predicted contradicted lines for story {story_id} :("
                )
                print("-----------")
                print()

        else:
            predicted_cont_error_lines = []
            predicted_contradicted_lines = []
            predicted_cont_error_lines_full = []
            predicted_contradicted_lines_full = []

        ground_truth_expl = agg_result["cont_error_expl"]

        eval_dataset.append(
            {
                "story_id": story_id,
                "story": story,
                "predicted_cont_error": predicted_cont_error,
                "ground_truth_cont_error": ground_truth_cont_error,
                "predicted_cont_error_lines": predicted_cont_error_lines,
                "predicted_cont_error_lines_full": predicted_cont_error_lines_full,
                "predicted_contradicted_lines": predicted_contradicted_lines,
                "predicted_contradicted_lines_full": predicted_contradicted_lines_full,
                "ground_truth_cont_error_lines": gt_cont_error_lines,
                "ground_truth_cont_error_lines_full": gt_cont_error_lines_full,
                "ground_truth_contradicted_lines": gt_contradicted_lines,
                "ground_truth_contradicted_lines_full": gt_contradicted_lines_full,
                "predicted_expl": predicted_expl,
                "ground_truth_expl": ground_truth_expl,
                "num_reasoning_tokens": num_reasoning_tokens,
                "num_completion_tokens": num_completion_tokens,
                "num_prompt_tokens": num_prompt_tokens,
            }
        )

    return pd.DataFrame(eval_dataset)


def eval_cont_error_localization_on_example(row):

    if row["predicted_cont_error"] != row["ground_truth_cont_error"]:
        return {
            "full_correct": 0,
            "cont_error_lines_correct": 0,
            "contradicted_lines_correct": 0,
        }

    if row["predicted_cont_error"] == False:
        return {
            "full_correct": 1,
            "cont_error_lines_correct": 1,
            "contradicted_lines_correct": 1,
        }
    predicted_cont_error_lines = set(row["predicted_cont_error_lines_full"])
    ground_truth_cont_error_lines = set(row["ground_truth_cont_error_lines_full"])

    # Check if any predicted line is in ground truth lines
    cont_error_line_detected = bool(
        predicted_cont_error_lines.intersection(ground_truth_cont_error_lines)
    )

    predicted_contradicted_lines = set(row["predicted_contradicted_lines_full"])
    ground_truth_contradicted_lines = set(row["ground_truth_contradicted_lines_full"])

    # Check if any predicted line is in ground truth lines
    contradicted_line_detected = bool(
        predicted_contradicted_lines.intersection(ground_truth_contradicted_lines)
    )

    return {
        "full_correct": float(cont_error_line_detected and contradicted_line_detected),
        "cont_error_lines_correct": float(cont_error_line_detected),
        "contradicted_lines_correct": float(contradicted_line_detected),
    }


def eval_cont_error_localization_strict_on_example(row):

    if row["predicted_cont_error"] != row["ground_truth_cont_error"]:
        return {
            "full_correct": 0,
            "cont_error_lines_correct": 0,
            "contradicted_lines_correct": 0,
        }

    if row["predicted_cont_error"] == False:
        return {
            "full_correct": 1,
            "cont_error_lines_correct": 1,
            "contradicted_lines_correct": 1,
        }

    predicted_cont_error_lines = set(row["predicted_cont_error_lines_full"])
    ground_truth_cont_error_lines = set(row["ground_truth_cont_error_lines_full"])
    predicted_contradicted_lines = set(row["predicted_contradicted_lines_full"])
    ground_truth_contradicted_lines = set(row["ground_truth_contradicted_lines_full"])

    # Check if all predicted lines are in ground truth lines
    cont_error_line_detected = len(
        predicted_cont_error_lines
    ) != 0 and predicted_cont_error_lines.issubset(ground_truth_cont_error_lines)
    contradicted_line_detected = len(
        predicted_contradicted_lines
    ) != 0 and predicted_contradicted_lines.issubset(ground_truth_contradicted_lines)

    return {
        "full_correct": float(cont_error_line_detected and contradicted_line_detected),
        "cont_error_lines_correct": float(cont_error_line_detected),
        "contradicted_lines_correct": float(contradicted_line_detected),
    }


def eval_cont_error_localization(results, dataset, pos_only=False, strict=False):
    
    eval_dataset = construct_eval_dataset(results, dataset)
    
    if pos_only:
        eval_dataset = eval_dataset[eval_dataset["ground_truth_cont_error"] == 1]
    
    if strict:
        eval_func = eval_cont_error_localization_strict_on_example
    else:
        eval_func = eval_cont_error_localization_on_example

    eval_results = eval_dataset.apply(eval_func, axis=1)

    full_correct_col_name = "full_correct" if not strict else "full_correct_strict"
    cont_error_lines_correct_col_name = (
        "cont_error_lines_correct" if not strict else "cont_error_lines_correct_strict"
    )
    contradicted_lines_correct_col_name = (
        "contradicted_lines_correct"
        if not strict
        else "contradicted_lines_correct_strict"
    )

    eval_dataset_w_results = eval_dataset.copy()

    eval_dataset_w_results[full_correct_col_name] = eval_results.apply(
        lambda x: x["full_correct"]
    )
    eval_dataset_w_results[cont_error_lines_correct_col_name] = eval_results.apply(
        lambda x: x["cont_error_lines_correct"]
    )
    eval_dataset_w_results[contradicted_lines_correct_col_name] = eval_results.apply(
        lambda x: x["contradicted_lines_correct"]
    )

    return (
        eval_dataset_w_results,
        eval_dataset_w_results[full_correct_col_name].mean(),
        eval_dataset_w_results[cont_error_lines_correct_col_name].mean(),
        eval_dataset_w_results[contradicted_lines_correct_col_name].mean(),
    )
