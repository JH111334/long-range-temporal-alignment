"""Shared controlled vocabularies."""

from enum import StrEnum


class Era(StrEnum):
    BCE = "BCE"
    CE = "CE"


class Granularity(StrEnum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    INTERVAL = "interval"


class EvidenceClass(StrEnum):
    OFFICIAL = "official"
    PAPER = "paper"
    THIRD_PARTY = "third_party"
    INTERNAL_DECISION = "internal_decision"
    ASSUMPTION = "assumption"
    SYNTHETIC = "synthetic"
    SMOKE_TEST = "smoke_test"
    MEASURED = "measured"


class DataRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PROBE = "probe"


class AssetStatus(StrEnum):
    CANDIDATE = "candidate"
    HOLD = "hold"
    APPROVED = "approved"
    REJECTED = "rejected"
