"""Clustered McNemar Test and Cluster-Bootstrap Inference for Correlated Temporal Data.

Implements:
1. Standard paired McNemar test with continuity correction and exact binomial test.
2. Clustered McNemar test (Obuchowski 1998 / Eliasziw & Donner 1991) to account for
   intraclass correlation within shared source records or event clusters.
3. Cluster-level bootstrap 95% confidence intervals for paired accuracy differences.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ClusteredMcNemarResult:
    """Statistical summary of paired comparison with cluster adjustment."""
    contingency_table: Dict[str, int]  # a, b, c, d, total
    accuracy_1: float
    accuracy_2: float
    delta_accuracy: float
    discordant_b: int  # Model 1 correct, Model 2 wrong
    discordant_c: int  # Model 1 wrong, Model 2 correct
    mcnemar_statistic: float
    p_value_standard: float
    p_value_exact: Optional[float]
    clustered_statistic: float
    p_value_clustered: float
    cluster_count: int
    intraclass_correlation_icc: float
    bootstrap_ci_95: Tuple[float, float]
    ci_excludes_zero: bool
    verdict: str  # SIGNIFICANT_GAIN, SIGNIFICANT_LOSS, EQUIVALENT, INCONCLUSIVE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normal_cdf(x: float) -> float:
    """Approximation of standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _chi2_sf_1df(x: float) -> float:
    """Survival function (p-value) for Chi-squared distribution with 1 degree of freedom."""
    if x <= 0:
        return 1.0
    # For 1 degree of freedom: P(Chi2 >= x) = 2 * (1 - Phi(sqrt(x)))
    z = math.sqrt(x)
    return 2.0 * (1.0 - _normal_cdf(z))


def _exact_binomial_p(b: int, c: int) -> float:
    """Two-sided exact binomial p-value under null hypothesis p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Sum binomial probabilities from 0 to k
    cum_p = 0.0
    for i in range(k + 1):
        cum_p += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * cum_p)


def compute_clustered_bootstrap_ci(
    cluster_differences: Dict[str, Tuple[int, int]],  # cluster_id -> (sum_d_k, m_k)
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute empirical percentile 95% bootstrap CI by resampling whole clusters."""
    import random
    rng = random.Random(seed)
    clusters = list(cluster_differences.keys())
    k = len(clusters)
    if k == 0:
        return (0.0, 0.0)

    total_n = sum(m for _, m in cluster_differences.values())
    if total_n == 0:
        return (0.0, 0.0)

    bootstrap_deltas = []
    for _ in range(n_bootstraps):
        resampled_clusters = [rng.choice(clusters) for _ in range(k)]
        sum_d = sum(cluster_differences[c][0] for c in resampled_clusters)
        sum_m = sum(cluster_differences[c][1] for c in resampled_clusters)
        if sum_m > 0:
            bootstrap_deltas.append(sum_d / sum_m)

    if not bootstrap_deltas:
        return (0.0, 0.0)

    bootstrap_deltas.sort()
    lower_idx = int((alpha / 2.0) * len(bootstrap_deltas))
    upper_idx = int((1.0 - alpha / 2.0) * len(bootstrap_deltas))
    upper_idx = min(upper_idx, len(bootstrap_deltas) - 1)

    return (round(bootstrap_deltas[lower_idx], 6), round(bootstrap_deltas[upper_idx], 6))


def compute_clustered_mcnemar(
    records_model1: List[Dict[str, Any]],
    records_model2: List[Dict[str, Any]],
    cluster_field: str = "source_record_id",
) -> ClusteredMcNemarResult:
    """Perform matched paired comparison with ordinary and clustered McNemar tests."""
    # Match examples by example_id
    map2 = {r["example_id"]: r for r in records_model2 if "example_id" in r}
    
    paired_items: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for r1 in records_model1:
        ex_id = r1.get("example_id")
        if ex_id in map2:
            paired_items.append((r1, map2[ex_id]))

    total = len(paired_items)
    if total == 0:
        raise ValueError("Zero overlapping examples found between compared prediction sets.")

    # Contingency table
    a = 0  # Both correct
    b = 0  # M1 correct, M2 incorrect
    c = 0  # M1 incorrect, M2 correct
    d = 0  # Both incorrect

    # Cluster grouping: cluster_id -> list of d_ij (c1 - c2)
    clusters: Dict[str, List[int]] = {}

    for r1, r2 in paired_items:
        c1 = 1 if r1.get("is_correct") else 0
        c2 = 1 if r2.get("is_correct") else 0
        diff = c1 - c2

        if c1 == 1 and c2 == 1:
            a += 1
        elif c1 == 1 and c2 == 0:
            b += 1
        elif c1 == 0 and c2 == 1:
            c += 1
        else:
            d += 1

        clust_id = str(r1.get(cluster_field) or r1.get("event_id") or r1.get("temporal_cluster_id") or "DEFAULT_CLUSTER")
        clusters.setdefault(clust_id, []).append(diff)

    acc1 = (a + b) / total
    acc2 = (a + c) / total
    delta_acc = acc1 - acc2

    # Standard McNemar with Edwards continuity correction
    discordant_sum = b + c
    if discordant_sum == 0:
        mcnemar_stat = 0.0
        p_std = 1.0
        p_exact = 1.0
    else:
        mcnemar_stat = (abs(b - c) - 1.0) ** 2 / discordant_sum if abs(b - c) > 1.0 else 0.0
        p_std = _chi2_sf_1df(mcnemar_stat)
        p_exact = _exact_binomial_p(b, c) if discordant_sum < 30 else None

    # Clustered McNemar (Obuchowski variance adjustment)
    num_clusters = len(clusters)
    cluster_diff_sums: Dict[str, Tuple[int, int]] = {}
    sum_d_all = b - c

    d_k_list = []
    m_k_list = []
    for c_id, diffs in clusters.items():
        d_k = sum(diffs)
        m_k = len(diffs)
        d_k_list.append(d_k)
        m_k_list.append(m_k)
        cluster_diff_sums[c_id] = (d_k, m_k)

    # Clustered variance calculation
    if num_clusters > 1:
        mean_d = sum_d_all / total
        v_cluster = 0.0
        for d_k, m_k in zip(d_k_list, m_k_list):
            expected_d_k = m_k * mean_d
            v_cluster += (d_k - expected_d_k) ** 2
        v_cluster *= (num_clusters / (num_clusters - 1.0))
        
        # Intraclass correlation approximation
        mean_m = total / num_clusters
        total_var = discordant_sum - (sum_d_all ** 2 / total)
        if total_var > 0:
            icc = (v_cluster / total - 1.0) / max(mean_m - 1.0, 1.0)
            icc = max(-1.0, min(1.0, icc))
        else:
            icc = 0.0
    else:
        v_cluster = float(discordant_sum)
        icc = 0.0

    se_cluster = math.sqrt(max(v_cluster, 1e-12))
    z_cluster = sum_d_all / se_cluster if se_cluster > 0 else 0.0
    chi2_cluster = z_cluster ** 2
    p_clustered = _chi2_sf_1df(chi2_cluster)

    # Bootstrap 95% CI
    ci_95 = compute_clustered_bootstrap_ci(cluster_diff_sums)
    ci_excludes_zero = (ci_95[0] > 0 and ci_95[1] > 0) or (ci_95[0] < 0 and ci_95[1] < 0)

    # Determine scientific verdict
    if p_clustered < 0.05 and ci_excludes_zero:
        verdict = "SIGNIFICANT_GAIN" if delta_acc > 0 else "SIGNIFICANT_LOSS"
    elif p_clustered >= 0.05 and abs(delta_acc) < 0.02 and (ci_95[1] - ci_95[0]) < 0.05:
        verdict = "EQUIVALENT"
    else:
        verdict = "INCONCLUSIVE"

    return ClusteredMcNemarResult(
        contingency_table={"a": a, "b": b, "c": c, "d": d, "total": total},
        accuracy_1=round(acc1, 6),
        accuracy_2=round(acc2, 6),
        delta_accuracy=round(delta_acc, 6),
        discordant_b=b,
        discordant_c=c,
        mcnemar_statistic=round(mcnemar_stat, 6),
        p_value_standard=round(p_std, 6),
        p_value_exact=round(p_exact, 6) if p_exact is not None else None,
        clustered_statistic=round(chi2_cluster, 6),
        p_value_clustered=round(p_clustered, 6),
        cluster_count=num_clusters,
        intraclass_correlation_icc=round(icc, 6),
        bootstrap_ci_95=ci_95,
        ci_excludes_zero=ci_excludes_zero,
        verdict=verdict,
    )
