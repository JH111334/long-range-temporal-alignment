"""Domain-specific errors."""


class TemporalAlignmentError(ValueError):
    """Base error for invalid temporal-alignment inputs."""


class ContractError(TemporalAlignmentError):
    """Raised when a data contract is incomplete or inconsistent."""


class EvidenceGateError(TemporalAlignmentError):
    """Raised when an external asset has not passed its evidence gate."""


class ConfigurationError(TemporalAlignmentError):
    """Raised when an experiment configuration is unsupported."""
