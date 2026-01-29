#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a strict GSM8K answer checker.\n"
    "Your task is ONLY to judge whether the FINAL numeric answer in model_output "
    "matches the reference_answer.\n\n"
    "Rules:\n"
    "- Extract the final numeric answer from both sides.\n"
    "- Ignore units, commas, currency symbols, and whitespace.\n"
    "- Accept equivalent values (e.g., 2 == 2.0, 0.5 == 1/2).\n"
    "- If multiple numbers appear, use the final stated answer.\n"
    "- If no clear final answer can be extracted, return same=false.\n\n"
    "Output MUST be valid JSON and NOTHING else.\n"
    "JSON format:\n"
    "{\n"
    '  "same": true | false\n'
    "}"
)


def load_json(path: str) -> List[Dict[str, Any]]:
    """Robust loader:

    Supports:
      1) JSON list: [ {...}, {...} ]
      2) JSON dict wrapper: {"data"/"results"/...: [ ... ]} (and one-level nested dict)
      3) JSONL: each line is a JSON object
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError("Empty input file.")

    # 1) Try standard JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj

        if isinstance(obj, dict):
            common_keys = [
                "data",
                "results",
                "records",
                "samples",
                "items",
                "eval",
                "evaluations",
                "outputs",
                "predictions",
            ]

            for k in common_keys:
                if k in obj and isinstance(obj[k], list):
                    return obj[k]

            # One-level nested dict
            for _, v in obj.items():
                if isinstance(v, dict):
                    for k in common_keys:
                        if k in v and isinstance(v[k], list):
                            return v[k]
    except json.JSONDecodeError:
        pass

    # 2) JSONL fallback
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSONL at line {i}: {e}") from e
            if isinstance(rec, dict):
                data.append(rec)
            else:
                # If a line is not an object, still store it as dict
                data.append({"value": rec})

    if not data:
        raise ValueError("Could not parse input as JSON list/dict or JSONL.")
    return data


def extract_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object substring from text and parse it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def create_client(api_key: str, base_url: Optional[str] = None) -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


_thread_local = threading.local()


def get_thread_client(api_key: str, base_url: Optional[str]) -> OpenAI:
    """Create one OpenAI client per thread to avoid any potential shared-state issues."""
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = create_client(api_key=api_key, base_url=base_url)
        _thread_local.client = client
    return client



def call_gpt_judge(
    api_key: str,
    base_url: Optional[str],
    ref: str,
    pred: str,
    model: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    sleep_base: float = 1.0,
) -> bool:
    client = get_thread_client(api_key=api_key, base_url=base_url)

    user_prompt = (
        f"reference_answer:\n{ref}\n\n"
        f"model_output:\n{pred}\n\n"
        "Return JSON only."
    )

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            result = extract_json(text)
            return bool(result.get("same", False))
        except Exception as e:
            last_err = e
            time.sleep(sleep_base * (2 ** attempt))

    raise RuntimeError(f"Judge failed after retries: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to GSM8K result JSON/JSONL")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Judge model name")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N samples")
    parser.add_argument("--out", default=None, help="Optional: save per-sample results (jsonl)")
    parser.add_argument("--api-key", default="[your_api]", help="API key (or set OPENAI_API_KEY)")
    parser.add_argument("--base-url", default="[your_url]", help="Base URL (e.g. https://xxx/v1)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Provide --api-key or set OPENAI_API_KEY.")

    data = load_json(args.input)
    if args.limit:
        data = data[: args.limit]

    correct = 0
    total = 0

    fout = open(args.out, "w", encoding="utf-8") if args.out else None

    try:
        # Multi-threaded judging
        def _task(item: Dict[str, Any]) -> Dict[str, Any]:
            ref = str(item.get("goal", item.get("answer", "")))
            pred = str(item.get("output", item.get("prediction", item.get("pred", ""))))
            rid = item.get("id", None)
            try:
                same = call_gpt_judge(
                    api_key=api_key,
                    base_url=args.base_url,
                    ref=ref,
                    pred=pred,
                    model=args.model,
                    temperature=args.temperature,
                )
                return {"id": rid, "same": bool(same), "error": None}
            except Exception as e:
                # Mark as incorrect on error, keep error message for inspection
                return {"id": rid, "same": False, "error": str(e)}

        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futures = [ex.submit(_task, item) for item in data]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
                rec = fut.result()

                total += 1
                if rec.get("same"):
                    correct += 1

                if fout:
                    # Save error field too (if any)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        acc = correct / total if total else 0.0
        print(f"\nTotal: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {acc:.4%}")

    finally:
        if fout:
            fout.close()


if __name__ == "__main__":
    main()