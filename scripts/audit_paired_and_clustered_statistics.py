"""CLI for Paired and Clustered Statistical Analysis between two evaluated models/conditions.

Computes:
1. Matched 2x2 contingency table (a, b, c, d).
2. Standard McNemar test with continuity correction and exact binomial p-value.
3. Clustered McNemar test with intraclass correlation adjustment.
4. Cluster-bootstrap 95% confidence intervals.
5. Generates formal PAIRED_STATISTICS_RECEIPT.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from temporal_alignment.causal_evaluation.clustered_mcnemar import (
    ClusteredMcNemarResult,
    compute_clustered_mcnemar,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Paired and Clustered Statistics")
    parser.add_argument("--m1_predictions", type=str, required=True, help="Path to Model 1 RAW_PREDICTIONS.jsonl")
    parser.add_argument("--m2_predictions", type=str, required=True, help="Path to Model 2 RAW_PREDICTIONS.jsonl")
    parser.add_argument("--m1_label", type=str, default="Model_1", help="Display name for Model 1")
    parser.add_argument("--m2_label", type=str, default="Model_2", help="Display name for Model 2")
    parser.add_argument("--cluster_field", type=str, default="source_record_id", help="Field used for clustering")
    parser.add_argument("--output_dir", type=str, default="artifacts/evaluation/statistics", help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1_path = Path(args.m1_predictions)
    if not p1_path.is_absolute():
        p1_path = _REPO_ROOT / p1_path
    p2_path = Path(args.m2_predictions)
    if not p2_path.is_absolute():
        p2_path = _REPO_ROOT / p2_path

    print(f"Loading predictions for {args.m1_label} from {p1_path}...")
    recs1 = load_jsonl(p1_path)
    print(f"Loading predictions for {args.m2_label} from {p2_path}...")
    recs2 = load_jsonl(p2_path)

    res: ClusteredMcNemarResult = compute_clustered_mcnemar(
        records_model1=recs1,
        records_model2=recs2,
        cluster_field=args.cluster_field,
    )

    t = res.contingency_table
    receipt = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "m1_label": args.m1_label,
            "m1_predictions_file": str(p1_path),
            "m2_label": args.m2_label,
            "m2_predictions_file": str(p2_path),
            "cluster_field": args.cluster_field,
        },
        "contingency_table": t,
        "metrics": {
            "m1_accuracy": res.accuracy_1,
            "m2_accuracy": res.accuracy_2,
            "delta_accuracy_m1_minus_m2": res.delta_accuracy,
            "discordant_b": res.discordant_b,
            "discordant_c": res.discordant_c,
        },
        "hypothesis_tests": {
            "standard_mcnemar_chi2": res.mcnemar_statistic,
            "standard_mcnemar_p_value": res.p_value_standard,
            "exact_binomial_p_value": res.p_value_exact,
            "clustered_mcnemar_chi2": res.clustered_statistic,
            "clustered_mcnemar_p_value": res.p_value_clustered,
            "intraclass_correlation_icc": res.intraclass_correlation_icc,
            "cluster_count": res.cluster_count,
        },
        "bootstrap_inference": {
            "bootstrap_ci_95": res.bootstrap_ci_95,
            "ci_excludes_zero": res.ci_excludes_zero,
        },
        "scientific_verdict": res.verdict,
    }

    receipt_path = out_dir / f"PAIRED_STATS_{args.m1_label}_VS_{args.m2_label}.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)

    print("\n================ PAIRED STATISTICAL COMPARISON ================")
    print(f"Comparison: {args.m1_label} vs {args.m2_label} (N = {t['total']})")
    print(f"Accuracy: {args.m1_label} = {res.accuracy_1:.4f} | {args.m2_label} = {res.accuracy_2:.4f}")
    print(f"Delta: {res.delta_accuracy:+.4f} (Discordant: b={res.discordant_b}, c={res.discordant_c})")
    print(f"Standard McNemar p: {res.p_value_standard:.6f} | Exact Binomial p: {res.p_value_exact}")
    print(f"Clustered McNemar p: {res.p_value_clustered:.6f} (Clusters: {res.cluster_count}, ICC: {res.intraclass_correlation_icc:.4f})")
    print(f"Cluster-Bootstrap 95% CI: [{res.bootstrap_ci_95[0]:+.4f}, {res.bootstrap_ci_95[1]:+.4f}]")
    print(f"CI Excludes Zero: {res.ci_excludes_zero}")
    print(f"Scientific Verdict: {res.verdict}")
    print(f"Receipt written to: {receipt_path}")
    print("===============================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
