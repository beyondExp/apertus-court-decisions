"""Load the published head and recompute the official scores from predictions.

A live decision still needs the last hidden states of frozen
swiss-ai/Apertus-v1.5-8B. This entry point does not download those weights.
It checks that the head runs, then rebuilds accuracy from the saved predictions
so a clean checkout can show the result.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheaf_head import load_head


ROOT = Path(__file__).resolve().parents[1]


def interface_check(head) -> None:
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, 4, 4096, generator=generator)
    state_mask = torch.ones(1, 4, dtype=torch.bool)
    option = torch.randn(1, 2, 3, 4096, generator=generator)
    option_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    with torch.no_grad():
        logits, _best = head.glue(hidden, state_mask, option, option_mask)
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise SystemExit("head forward failed")
    print("head forward ok (shape check on random states, not a court decision)")


def recompute(path: Path) -> dict:
    scores = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            bucket = scores.setdefault((row["split"], row["task"]), {"n": 0, "correct": 0})
            bucket["n"] += 1
            bucket["correct"] += int(row["pred"] == row["gold"])
    return scores


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    reader = os.environ.get("LLM_NAME") or "swiss-ai/Apertus-v1.5-8B"
    print(f"reader {reader}")
    if os.environ.get("LLM_BASE_URL"):
        print("LLM_BASE_URL is set and unused: this head scores hidden states, not chat text")
    head = load_head(ROOT / "checkpoint" / "head.pt")
    n_params = sum(p.numel() for p in head.parameters())
    print(f"sheaf head parameters {n_params}")
    interface_check(head)

    metrics = json.loads((ROOT / "data" / "metrics.json").read_text(encoding="utf-8"))
    fresh = recompute(ROOT / "data" / "predictions.jsonl")
    print()
    print(f"{'split':<10} {'task':<12} {'n':>7} {'accuracy':>10} {'majority':>10}")
    for split_name, tasks in metrics["splits"].items():
        for task_name, block in tasks.items():
            got = fresh[(split_name, task_name)]
            acc = got["correct"] / got["n"]
            if got["n"] != block["n"] or abs(acc - block["accuracy"]) > 1e-9:
                raise SystemExit(f"prediction file does not match metrics for {split_name} {task_name}")
            print(
                f"{split_name:<10} {task_name:<12} {block['n']:7d} "
                f"{pct(block['accuracy']):>10} {pct(block['majority_accuracy']):>10}"
            )
    law = metrics["splits"]["test"]["law_area"]
    gate = next(row for row in law["confidence"] if row["threshold"] == 0.9)
    print()
    print(
        "test legal area at confidence >= 0.90: "
        f"answers {pct(gate['coverage'])} of cases, accuracy {pct(gate['accuracy'])}"
    )
    judgment = metrics["splits"]["test"]["judgment"]
    print(
        "test outcome accuracy "
        f"{pct(judgment['accuracy'])}, majority baseline {pct(judgment['majority_accuracy'])}"
    )
    print("predictions match the published metrics")


if __name__ == "__main__":
    main()
