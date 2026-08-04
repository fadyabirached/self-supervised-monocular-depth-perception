#!/usr/bin/env python3
"""Turn one or more metrics_logger.py CSVs into the README comparison table.

Usage:
    scripts/summarize_metrics.py depth_metrics.csv yolo_metrics.csv

Prints a markdown table (success rate, reacted rate, mean reaction time,
collision rate, encounter count) grouped by the ``method`` column, plus a
few sanity flags for suspicious rows. No ROS or third-party dependencies,
so it runs anywhere Python does, including outside the container.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def to_bool(value: str) -> bool:
    return value.strip() == "1"


def summarize(rows: list[dict]) -> dict[str, dict]:
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    summary = {}
    for method, method_rows in by_method.items():
        n = len(method_rows)
        successes = sum(to_bool(r["success"]) for r in method_rows)
        collisions = sum(to_bool(r["collision"]) for r in method_rows)
        reacted = [r for r in method_rows if to_bool(r["reacted"])]
        reaction_times = [float(r["reaction_time"]) for r in reacted
                           if float(r["reaction_time"]) >= 0]

        neither = n - successes - collisions
        summary[method] = {
            "n": n,
            "success_rate": successes / n if n else 0.0,
            "collision_rate": collisions / n if n else 0.0,
            "reacted_rate": len(reacted) / n if n else 0.0,
            "mean_reaction_s": statistics.mean(reaction_times) if reaction_times else None,
            "neither_marked": neither,
        }
    return summary


def print_table(summary: dict[str, dict]) -> None:
    print("| Method | Encounters | Success rate | Collision rate | Reacted | Mean reaction time |")
    print("|---|---|---|---|---|---|")
    for method in sorted(summary):
        s = summary[method]
        reaction = f"{s['mean_reaction_s']:.2f}s" if s["mean_reaction_s"] is not None else "n/a"
        print(
            f"| {method} | {s['n']} | {s['success_rate']:.0%} | "
            f"{s['collision_rate']:.0%} | {s['reacted_rate']:.0%} | {reaction} |"
        )

    print()
    for method in sorted(summary):
        s = summary[method]
        if s["n"] < 10:
            print(f"Note: only {s['n']} encounters logged for '{method}', treat this as a small sample.")
        if s["neither_marked"]:
            print(f"Note: {s['neither_marked']} '{method}' rows are marked neither success nor "
                  "collision (an 'n' with no 's'/'c'), check the raw CSV before trusting the rates above.")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    rows = load_rows(argv)
    if not rows:
        print("No rows found in the given file(s).", file=sys.stderr)
        return 1

    print_table(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
