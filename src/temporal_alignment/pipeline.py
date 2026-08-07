"""Deterministic Tier-0 pipeline over governed synthetic examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .calendar import published_sexagenary_index
from .encoders import encoder_from_name
from .evaluation import alias_collision_rate, temporal_distance_correlation
from .io import canonical_sha256, file_sha256, load_examples, load_json, unique_ids, write_json
from .objectives import normalized_alignment_loss, published_ticktack_loss
from .splits import split_examples


def run_smoke_pipeline(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Run representation diagnostics without external models or datasets."""
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_json(config_file)
    dataset_path = project_root / config["dataset_path"]
    examples = load_examples(dataset_path)
    if not unique_ids(examples):
        raise ValueError("example_id values must be unique")

    splits = split_examples(examples, seed=config["split_seed"])
    years = [example.event_time.start for example in examples]
    labels = [published_sexagenary_index(year) for year in years]
    encoder_reports: list[dict[str, Any]] = []

    for encoder_name in config["encoders"]:
        encoder = encoder_from_name(encoder_name)
        vectors = [encoder.encode(year) for year in years]
        published_loss = published_ticktack_loss(vectors, labels, delta=1.0)
        normalized_loss = normalized_alignment_loss(vectors, labels, delta=0.5)
        encoder_reports.append(
            {
                "encoder_name": encoder_name,
                "encoder_id": encoder.encoder_id,
                "evidence_class": encoder.evidence_class,
                "dimension": len(vectors[0]),
                "alias_collision_rate_60": alias_collision_rate(years, encoder.encode),
                "temporal_distance_correlation": temporal_distance_correlation(
                    years, encoder.encode
                ),
                "published_loss_delta_1": published_loss.total,
                "normalized_loss_delta_0_5": normalized_loss.total,
                "intra_pairs": normalized_loss.intra_pairs,
                "inter_pairs": normalized_loss.inter_pairs,
            }
        )

    result = {
        "run_id": f"smoke-{canonical_sha256(config)[:12]}",
        "evidence_class": "smoke_test",
        "config_sha256": canonical_sha256(config),
        "data_sha256": file_sha256(dataset_path),
        "dataset_id": config["dataset_id"],
        "example_count": len(examples),
        "split_counts": {name: len(items) for name, items in sorted(splits.items())},
        "encoders": encoder_reports,
        "warnings": [
            "Dữ liệu tổng hợp chỉ kiểm tra hợp đồng, không đo năng lực LLM.",
            "Published Ticktack giữ nguyên va chạm 60 năm từ phương trình đã in.",
        ],
    }
    write_json(output_path, result)
    return result
