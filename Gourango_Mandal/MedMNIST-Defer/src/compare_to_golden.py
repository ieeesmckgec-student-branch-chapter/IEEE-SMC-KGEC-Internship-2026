#!/usr/bin/env python3
"""Compare a fresh benchmark run against the committed reference results.

Small numerical drift is expected across GPU models and library versions, so a
tolerance is applied rather than requiring bit-identical output.

Usage:
    python scripts/compare_to_golden.py --candidate results/benchmark_full.json
"""
import argparse
import sys
from pathlib import Path

from common import load_json
from config import GOLDEN_DIR

DEFAULT_TOLERANCE = 0.05


def compare(golden: dict, candidate: dict, tolerance: float) -> tuple[list[str], list[str]]:
    matches, mismatches = [], []
    shared = sorted(set(golden["per_dataset"]) & set(candidate["per_dataset"]))

    for name in shared:
        reference = golden["per_dataset"][name]["policies"]["entropy"]["acc_at_coverage"]
        fresh = candidate["per_dataset"][name]["policies"]["entropy"]["acc_at_coverage"]
        for key, expected in reference.items():
            actual = fresh.get(key)
            if actual is None:
                mismatches.append(f"{name} {key}: missing in candidate")
                continue
            difference = abs(actual - expected)
            line = f"{name:15s} {key:9s} golden={expected:.4f} run={actual:.4f} Δ={difference:+.4f}"
            (matches if difference <= tolerance else mismatches).append(line)

    reference_gain = golden["aggregate"]["mean_gain_at_50pct_entropy"]
    fresh_gain = candidate["aggregate"]["mean_gain_at_50pct_entropy"]
    line = (
        f"{'AGGREGATE':15s} mean gain @50% golden={reference_gain:.4f} "
        f"run={fresh_gain:.4f} Δ={fresh_gain - reference_gain:+.4f}"
    )
    (matches if abs(fresh_gain - reference_gain) <= tolerance else mismatches).append(line)

    return matches, mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--golden", default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    candidate = load_json(Path(args.candidate))
    golden_path = (
        Path(args.golden)
        if args.golden
        else GOLDEN_DIR / f"benchmark_{candidate.get('mode', 'full')}.json"
    )
    if not golden_path.exists():
        print(f"No golden file at {golden_path}", file=sys.stderr)
        return 1

    golden = load_json(golden_path)
    matches, mismatches = compare(golden, candidate, args.tolerance)

    print(f"Golden:    {golden_path}")
    print(f"Candidate: {args.candidate}")
    print(f"Tolerance: +/-{args.tolerance:.3f} absolute accuracy\n")
    for line in mismatches:
        print(f"[DIFF] {line}")
    print(f"\n{len(matches)} comparisons within tolerance, {len(mismatches)} outside.")

    if mismatches:
        print(
            "\nDifferences of a few points are normal on different hardware. "
            "Large gaps usually mean a different seed, epoch count, or data mode."
        )
    return 0 if not mismatches else 2


if __name__ == "__main__":
    sys.exit(main())
