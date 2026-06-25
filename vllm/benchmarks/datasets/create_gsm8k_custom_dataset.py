# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a CustomDataset JSONL file from GSM8K.

The output can be passed to ``vllm bench serve --dataset-name custom``.

Example:
    .venv/bin/python vllm/benchmarks/datasets/create_gsm8k_custom_dataset.py \
        --split test \
        --output benchmarks/gsm8k_test_custom.jsonl \
        --output-len 1024
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections.abc import Iterable
from pathlib import Path


GSM8K_URLS = {
    "train": (
        "https://raw.githubusercontent.com/openai/grade-school-math/master/"
        "grade_school_math/data/train.jsonl"
    ),
    "test": (
        "https://raw.githubusercontent.com/openai/grade-school-math/master/"
        "grade_school_math/data/test.jsonl"
    ),
}

PROMPT_TEMPLATES = {
    "boxed": (
        "{question}\nPlease reason step by step, and put your final answer "
        "within \\boxed{{}}."
    ),
    "qa": "Question: {question}\nAnswer:",
    "plain": "{question}",
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield json.loads(line)


def download_gsm8k(split: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"gsm8k_{split}.jsonl"
    if path.exists():
        return path

    url = GSM8K_URLS[split]
    print(f"Downloading {url} -> {path}")
    with urllib.request.urlopen(url) as response:
        path.write_bytes(response.read())
    return path


def convert_gsm8k(
    *,
    input_path: Path,
    output_path: Path,
    output_len: int,
    prompt_template: str,
    max_samples: int | None,
    shuffle: bool,
    seed: int,
) -> int:
    rows = list(read_jsonl(input_path))
    if shuffle:
        random.Random(seed).shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]

    template = PROMPT_TEMPLATES[prompt_template]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            question = row["question"]
            record = {
                "prompt": template.format(question=question),
                "output_tokens": output_len,
                "answer": row.get("answer"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GSM8K into custom JSONL for vLLM benchmarks."
    )
    parser.add_argument(
        "--split",
        choices=sorted(GSM8K_URLS),
        default="test",
        help="GSM8K split to download when --input is not provided.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional local GSM8K JSONL path. If omitted, downloads --split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output custom JSONL path. Defaults to "
            "benchmarks/gsm8k_<split>_custom.jsonl."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("benchmarks/.cache/gsm8k"),
        help="Directory used for downloaded raw GSM8K files.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1024,
        help="output_tokens value written for each request.",
    )
    parser.add_argument(
        "--prompt-template",
        choices=sorted(PROMPT_TEMPLATES),
        default="boxed",
        help="Prompt format to use.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of examples to write.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle examples before applying --max-samples.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    if input_path is None:
        input_path = download_gsm8k(args.split, args.cache_dir)
    output_path = args.output
    if output_path is None:
        output_path = Path(f"benchmarks/gsm8k_{args.split}_custom.jsonl")

    count = convert_gsm8k(
        input_path=input_path,
        output_path=output_path,
        output_len=args.output_len,
        prompt_template=args.prompt_template,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(f"Wrote {count} GSM8K examples to {output_path}")


if __name__ == "__main__":
    main()
