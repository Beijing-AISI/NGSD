import json
import os
from typing import List, Dict, Any
from statistics import mean, median


def load_json(path: str) -> List[Dict[str, Any]]:
    """加载 JSON 文件，返回 list[dict]"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        for key in ["data", "items", "records"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported JSON format in {path}")


def describe(name: str, values: List[int]):
    """打印一组长度的统计量"""
    if not values:
        print(f"{name}: No data\n")
        return

    print(f"{name}:")
    print(f"  count      = {len(values)}")
    print(f"  mean       = {mean(values):.2f}")
    print(f"  median     = {median(values)}")
    print(f"  min        = {min(values)}")
    print(f"  max        = {max(values)}")

    total = len(values)
    over_500 = sum(1 for v in values if v > 500)
    over_1000 = sum(1 for v in values if v > 1000)
    over_1500 = sum(1 for v in values if v > 1500)
    over_2000 = sum(1 for v in values if v > 2000)
    pct_500 = over_500 / total * 100
    pct_1000 = over_1000 / total * 100
    pct_1500 = over_1500 / total * 100
    pct_2000 = over_2000 / total * 100

    print(f"  > 500      = {over_500} ({pct_500:.2f}%)")
    print(f"  > 1000     = {over_1000} ({pct_1000:.2f}%)")
    print(f"  > 1500     = {over_1000} ({pct_1500:.2f}%)")
    print(f"  > 2000     = {over_1000} ({pct_2000:.2f}%)")
    print()


def stats_for_file(path: str, target_model: str | None = None):
    """
    对单个文件做统计：
    - 如果 target_model 不为 None，则只统计 target-model == target_model 的样本
    - 如果 target_model 为 None，则统计所有有 prompt 的样本
    """
    records = load_json(path)

    char_lengths: List[int] = []
    word_lengths: List[int] = []

    for item in records:
        # 如果要过滤模型
        if target_model is not None:
            model = item.get("target-model") or item.get("target_model")
            if model != target_model:
                continue
        if "alpaca" not in path:
            prompt = item.get("prompt")
        else:
            prompt = item.get("output")
        if not isinstance(prompt, str):
            continue

        char_lengths.append(len(prompt))
        word_lengths.append(len(prompt.split()))

    print(f"================ {path} ================")
    if target_model is not None:
        print(f"Model filter: {target_model}\n")
    else:
        print("Model filter: <none> (all prompts)\n")

    if not char_lengths:
        print("No records found under this condition.\n")
        return

    describe("Character length", char_lengths)
    describe("Word length", word_lengths)


def main():
    # 有 target-model 的文件：只统计 vicuna
    files_with_model = [
        ("GCG.json", "vicuna"),
        ("PAIR.json", "vicuna"),
        ("AutoDAN.json", "vicuna"),
    ]

    # 没有 target-model 的文件：全量统计
    # 文件名按你的实际来改，比如 "jsm8k.json" / "gsm8k.json"
    files_no_model = [
        "gsm8k.json",
        "FalseReject_test.json",
        "alpaca_eval.json",
    ]

    # 有 model 过滤的那几个
    for fname, model in files_with_model:
        if not os.path.exists("../datasets/"+ fname):
            print(f"Warning: {fname} not found, skip.\n")
            continue
        stats_for_file("../datasets/"+ fname, target_model=model)

    # 只有 prompt 的这几个
    for fname in files_no_model:
        if not os.path.exists("../datasets/"+ fname):
            print(f"Warning: {fname} not found, skip.\n")
            continue
        stats_for_file("../datasets/"+ fname, target_model=None)


if __name__ == "__main__":
    main()