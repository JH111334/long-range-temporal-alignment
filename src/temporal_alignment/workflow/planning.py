"""Describe the research pipeline without pretending stages are implemented."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import _required
from ..errors import ContractError


@dataclass(frozen=True, slots=True)
class StageContract:
    """Declare one pipeline stage and its validation boundary."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    validations: tuple[str, ...]
    failure_mode: str
    owner: str

    def __post_init__(self) -> None:
        for name in ("name", "failure_mode", "owner"):
            _required(getattr(self, name), name)
        if not self.inputs or not self.outputs or not self.validations:
            raise ContractError("stage inputs, outputs, and validations must not be empty")


@dataclass(frozen=True, slots=True)
class ResearchWorkflowPlan:
    """Validate the minimum Vietnam-first research stage sequence."""

    primary_use_case: str
    stages: tuple[StageContract, ...]

    def assert_complete(self) -> None:
        """Reject plans that omit a required lifecycle boundary."""
        _required(self.primary_use_case, "primary_use_case")
        names = tuple(stage.name for stage in self.stages)
        if len(names) != len(set(names)):
            raise ContractError("workflow stage names must be unique")
        missing = [name for name in _REQUIRED_STAGE_NAMES if name not in names]
        if missing:
            raise ContractError(f"workflow missing required stages: {', '.join(missing)}")
        positions = [names.index(name) for name in _REQUIRED_STAGE_NAMES]
        if positions != sorted(positions):
            raise ContractError("workflow stages are out of required lifecycle order")


_REQUIRED_STAGE_NAMES = (
    "source_survey",
    "evidence_gate",
    "normalization",
    "split",
    "retrieval",
    "representation",
    "reasoning",
    "evaluation",
    "registration",
    "application",
)


def default_research_stages() -> tuple[StageContract, ...]:
    """Return stage contracts for planning and documentation only."""
    stage_rows = (
        (
            "source_survey",
            ("source_candidate",),
            ("registry_record",),
            ("official_url", "terms"),
        ),
        (
            "evidence_gate",
            ("registry_record",),
            ("approved_manifest",),
            ("revision", "checksum"),
        ),
        (
            "normalization",
            ("approved_manifest",),
            ("canonical_examples",),
            ("era", "interval"),
        ),
        (
            "split",
            ("canonical_examples",),
            ("frozen_splits",),
            ("group_isolation", "near_duplicate"),
        ),
        (
            "retrieval",
            ("frozen_splits",),
            ("ranked_evidence",),
            ("time_filter", "source_filter"),
        ),
        (
            "representation",
            ("ranked_evidence",),
            ("temporal_features",),
            ("alias", "finite_values"),
        ),
        (
            "reasoning",
            ("temporal_features",),
            ("raw_prediction",),
            ("resource_gate", "output_schema"),
        ),
        (
            "evaluation",
            ("raw_prediction",),
            ("metrics",),
            ("invalid_output", "held_out_slices"),
        ),
        ("registration", ("metrics",), ("run_artifact",), ("config_data_code_hashes",)),
        (
            "application",
            ("run_artifact",),
            ("answer_or_abstention",),
            ("evidence_ids", "rollback"),
        ),
    )
    return tuple(
        StageContract(
            name=name,
            inputs=inputs,
            outputs=outputs,
            validations=validations,
            failure_mode="fail_closed",
            owner="unassigned",
        )
        for name, inputs, outputs, validations in stage_rows
    )
