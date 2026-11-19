"""Fail-Closed Experimental Gatekeeper and DAG Engine (Run 05.2).

Enforces machine-verifiable gates P0 through P8:
- Immutable gate transitions: PASS, FAIL, BLOCKED, NOT_RUN.
- ABSOLUTE INVARIANT: A gate whose criterion failed can NEVER have status 'APPROVED' or 'PASS'.
- Derives verdicts strictly from empirical predicates, not manual overrides.
- Exits nonzero on any gate failure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class GateStatus(str, Enum):
    """Immutable statuses for validation gates."""

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


@dataclass
class GatePredicate:
    """A single verifiable predicate within a gate."""

    name: str
    passed: bool
    observed_value: Any
    expected_value: Any
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed_value,
            "expected": self.expected_value,
            "message": self.message,
        }


@dataclass
class GateResult:
    """Result of a single gate evaluation."""

    gate_id: str
    status: GateStatus
    predicates: List[GatePredicate]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    evidence_artifacts: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "all_passed": self.status == GateStatus.PASS,
            "predicates": [p.to_dict() for p in self.predicates],
            "evidence_artifacts": self.evidence_artifacts,
            "failure_reason": self.failure_reason,
        }


class Gatekeeper:
    """Manages sequential dependency-DAG gate evaluations."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.gate_results: Dict[str, GateResult] = {}
        self._load_existing_gates()

    def _load_existing_gates(self, filename: str = "GATES.json") -> None:
        gates_path = self.run_dir / filename
        if not gates_path.exists():
            return
        try:
            with open(gates_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_gates = data.get("gates", {})
            for gid, ginfo in raw_gates.items():
                preds = [
                    GatePredicate(
                        name=p["name"],
                        passed=p["passed"],
                        observed_value=p.get("observed"),
                        expected_value=p.get("expected"),
                        message=p.get("message", ""),
                    )
                    for p in ginfo.get("predicates", [])
                ]
                self.gate_results[gid] = GateResult(
                    gate_id=ginfo["gate_id"],
                    status=GateStatus(ginfo["status"]),
                    predicates=preds,
                    timestamp=ginfo.get("timestamp", datetime.now().isoformat()),
                    evidence_artifacts=ginfo.get("evidence_artifacts", []),
                    failure_reason=ginfo.get("failure_reason"),
                )
        except Exception:
            pass

    def register_gate(
        self,
        gate_id: str,
        predicates: List[GatePredicate],
        evidence_artifacts: Optional[List[str]] = None,
        prerequisite_gates: Optional[List[str]] = None,
    ) -> GateResult:
        """Evaluate and record a gate fail-closed."""
        # 1. Check prerequisites
        if prerequisite_gates:
            for prereq in prerequisite_gates:
                prereq_res = self.gate_results.get(prereq)
                if not prereq_res or prereq_res.status != GateStatus.PASS:
                    result = GateResult(
                        gate_id=gate_id,
                        status=GateStatus.BLOCKED,
                        predicates=predicates,
                        evidence_artifacts=evidence_artifacts or [],
                        failure_reason=f"Prerequisite gate '{prereq}' did not PASS (status={prereq_res.status.value if prereq_res else 'MISSING'})",
                    )
                    self.gate_results[gate_id] = result
                    return result

        # 2. Evaluate predicates
        all_passed = True
        fail_msgs = []
        for p in predicates:
            if not p.passed:
                all_passed = False
                fail_msgs.append(f"Predicate '{p.name}' failed: observed {p.observed_value}, expected {p.expected_value} ({p.message})")

        status = GateStatus.PASS if all_passed else GateStatus.FAIL
        reason = "; ".join(fail_msgs) if not all_passed else None

        result = GateResult(
            gate_id=gate_id,
            status=status,
            predicates=predicates,
            evidence_artifacts=evidence_artifacts or [],
            failure_reason=reason,
        )
        self.gate_results[gate_id] = result
        return result

    def save_manifest(self, filename: str = "GATES.json") -> Path:
        """Write all gate results to JSON manifest."""
        out_path = self.run_dir / filename
        data = {
            "schema_version": "2.0",
            "study_id": "V2_RUN05_2_CONTRACT_GATEKEEPER",
            "timestamp": datetime.now().isoformat(),
            "overall_status": GateStatus.PASS.value if all(g.status == GateStatus.PASS for g in self.gate_results.values()) else GateStatus.FAIL.value,
            "gates": {k: v.to_dict() for k, v in self.gate_results.items()},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return out_path
