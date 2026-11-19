"""Typed and versioned experiment data schemas and strict validation contracts.

Enforces explicit temporal fields, rejects magic default years (e.g. 2000, 1984),
and provides fail-closed parsing with sample ID and field-level provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class SchemaValidationError(ValueError):
    """Raised when an item fails contract schema validation."""

    def __init__(self, sample_id: str, field_name: str, message: str, filename: Optional[str] = None) -> None:
        file_ctx = f" in file '{filename}'" if filename else ""
        super().__init__(f"Validation error for sample '{sample_id}'{file_ctx} on field '{field_name}': {message}")
        self.sample_id = sample_id
        self.field_name = field_name
        self.filename = filename


class TemporalConditionType(str, Enum):
    """Enumeration of causal intervention conditions for inference."""

    C_TRUE = "C_TRUE"
    C_WRONG_YEAR = "C_WRONG_YEAR"
    C_INPUT_ZERO_FEATURE = "C_INPUT_ZERO_FEATURE"
    C_OUTPUT_ZERO_SEAM = "C_OUTPUT_ZERO_SEAM"
    C_SEAM_OFF = "C_SEAM_OFF"
    C_LORA_OFF = "C_LORA_OFF"
    C_BOTH_OFF = "C_BOTH_OFF"


@dataclass(frozen=True, slots=True)
class NWPRecord:
    """Next-Word Prediction corpus record."""

    example_id: str
    input_text: str
    target_text: str
    canonical_year: Optional[int] = None
    source_record_id: Optional[str] = None
    task_id: str = "NWP"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], filename: Optional[str] = None) -> "NWPRecord":
        # Check example_id (not id)
        example_id = data.get("example_id")
        if not example_id or not isinstance(example_id, str):
            raise SchemaValidationError(str(example_id or "UNKNOWN"), "example_id", "Missing or non-string example_id", filename)

        input_text = data.get("input")
        if input_text is None or not isinstance(input_text, str):
            raise SchemaValidationError(example_id, "input", "Missing or non-string input", filename)

        target_text = data.get("target")
        if target_text is None or not isinstance(target_text, str):
            raise SchemaValidationError(example_id, "target", "Missing or non-string target", filename)

        canonical_year = data.get("canonical_year")
        if canonical_year is not None and not isinstance(canonical_year, int):
            try:
                canonical_year = int(canonical_year)
            except (ValueError, TypeError):
                raise SchemaValidationError(example_id, "canonical_year", f"Invalid year: {canonical_year}", filename)

        return cls(
            example_id=example_id,
            input_text=input_text,
            target_text=target_text,
            canonical_year=canonical_year,
            source_record_id=data.get("source_record_id"),
            task_id=str(data.get("task_id", "NWP")),
        )


@dataclass(frozen=True, slots=True)
class RelationPair:
    """Temporal relation comparison pair (Yes/No target)."""

    sample_id: str
    question: str
    target: str
    year_eval: int
    year_base: int
    entity: Optional[str] = None
    source_record_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], filename: Optional[str] = None) -> "RelationPair":
        sample_id = str(data.get("sample_id") or data.get("pair_id") or data.get("example_id") or "")
        if not sample_id:
            raise SchemaValidationError("UNKNOWN", "sample_id", "Missing identifier", filename)

        question = data.get("question") or data.get("input")
        if not question or not isinstance(question, str):
            raise SchemaValidationError(sample_id, "question", "Missing or non-string question/input", filename)

        target = data.get("target")
        if not target or target not in ("Yes", "No"):
            raise SchemaValidationError(sample_id, "target", f"Target must be 'Yes' or 'No', got '{target}'", filename)

        # year_eval and year_base must be explicit ints
        y_eval = data.get("year_eval")
        if y_eval is None:
            raise SchemaValidationError(sample_id, "year_eval", "Field 'year_eval' is required; no magic defaults allowed", filename)
        try:
            year_eval = int(y_eval)
        except (ValueError, TypeError):
            raise SchemaValidationError(sample_id, "year_eval", f"Invalid year_eval: {y_eval}", filename)

        y_base = data.get("year_base")
        if y_base is None:
            raise SchemaValidationError(sample_id, "year_base", "Field 'year_base' is required; no magic defaults allowed", filename)
        try:
            year_base = int(y_base)
        except (ValueError, TypeError):
            raise SchemaValidationError(sample_id, "year_base", f"Invalid year_base: {y_base}", filename)

        return cls(
            sample_id=sample_id,
            question=question,
            target=target,
            year_eval=year_eval,
            year_base=year_base,
            entity=data.get("entity"),
            source_record_id=data.get("source_record_id"),
        )


@dataclass(frozen=True, slots=True)
class MCQARecord:
    """Multiple Choice Question Answering record (4 options, AST verified)."""

    sample_id: str
    question: str
    options: Dict[str, str]
    target: str
    year_eval: int
    year_base: int
    entity: str
    direction: str
    options_ast: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], filename: Optional[str] = None) -> "MCQARecord":
        sample_id = str(data.get("sample_id") or "")
        if not sample_id:
            raise SchemaValidationError("UNKNOWN", "sample_id", "Missing sample_id", filename)

        question = data.get("question")
        if not question or not isinstance(question, str):
            raise SchemaValidationError(sample_id, "question", "Missing or non-string question", filename)

        target = data.get("target")
        if not target or target not in ("a", "b", "c", "d"):
            raise SchemaValidationError(sample_id, "target", f"Target must be in ['a', 'b', 'c', 'd'], got '{target}'", filename)

        options = data.get("options")
        if not options or not isinstance(options, dict) or set(options.keys()) != {"a", "b", "c", "d"}:
            raise SchemaValidationError(sample_id, "options", "Options must be a dict with keys a, b, c, d", filename)

        y_eval = data.get("year_eval")
        if y_eval is None:
            raise SchemaValidationError(sample_id, "year_eval", "Field 'year_eval' is strictly required; no magic defaults allowed", filename)
        try:
            year_eval = int(y_eval)
        except (ValueError, TypeError):
            raise SchemaValidationError(sample_id, "year_eval", f"Invalid year_eval: {y_eval}", filename)

        y_base = data.get("year_base")
        if y_base is None:
            raise SchemaValidationError(sample_id, "year_base", "Field 'year_base' is strictly required; no magic defaults allowed", filename)
        try:
            year_base = int(y_base)
        except (ValueError, TypeError):
            raise SchemaValidationError(sample_id, "year_base", f"Invalid year_base: {y_base}", filename)

        entity = str(data.get("entity") or "")
        direction = str(data.get("direction") or "")

        return cls(
            sample_id=sample_id,
            question=question,
            options=dict(options),
            target=target,
            year_eval=year_eval,
            year_base=year_base,
            entity=entity,
            direction=direction,
            options_ast=data.get("options_ast"),
        )
