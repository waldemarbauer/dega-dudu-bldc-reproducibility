"""Structured exception taxonomy for repository boundary failures."""

from __future__ import annotations

from pathlib import Path


class TrustworthyAgentError(Exception):
    """Base class for project-specific errors.

    Purpose:
        Preserve structured failure categories without swallowing causes.
    Parameters:
        Message and optional exception details accepted by `Exception`.
    Return value:
        Exception instance.
    Raised exceptions:
        None during construction.
    Scientific assumptions:
        None.
    Side effects:
        None.
    Reproducibility implications:
        Stable exception types support deterministic failure reporting.
    """


class ConfigurationError(TrustworthyAgentError):
    """Configuration validation or resolution failed."""


class SchemaValidationError(TrustworthyAgentError):
    """Dataset schema validation failed explicitly."""


class DatasetIntegrityError(TrustworthyAgentError):
    """Pinned dataset integrity verification failed."""


class DataQualityError(TrustworthyAgentError):
    """Input data quality is insufficient for the requested operation."""


class LeakageError(TrustworthyAgentError):
    """A learned operation attempted to fit outside the training split."""


class StrategyResolutionError(TrustworthyAgentError):
    """A configured strategy could not be resolved through the registry."""


class StrategyExecutionError(TrustworthyAgentError):
    """A state strategy failed while producing structured state facts."""


class TransitionPolicyError(TrustworthyAgentError):
    """A transition policy produced invalid or unsupported evidence."""


class InvalidTransitionError(TrustworthyAgentError):
    """A transition target is not allowed by the active state profile."""


class SafetyViolationError(TrustworthyAgentError):
    """SafetyGuard rejected or overrode an unsafe transition/action."""


class AuditWriteError(TrustworthyAgentError):
    """Audit persistence failed and the system must fail closed."""


class ReplayDivergenceError(TrustworthyAgentError):
    """Replay did not reproduce stored decision evidence."""


class ReproducibilityError(TrustworthyAgentError):
    """A reproducibility check failed."""


class RequiredArtifactMissingError(ReproducibilityError):
    """A required generated artifact is unavailable.

    Purpose:
        Report missing production artifacts with enough context for a workflow
        runner or maintainer to regenerate them deliberately.
    Parameters:
        artifact_name: Logical artifact identity.
        expected_path: Path where the artifact was expected.
        producing_stage: Pipeline stage responsible for creating the artifact.
        canonical_command: Canonical repository command that produces it.
    Return value:
        Exception instance with structured public attributes.
    Raised exceptions:
        None during construction.
    Scientific assumptions:
        Missing artifacts must fail closed. They are not generated implicitly
        because doing so could hide split, model, or statistical provenance
        changes.
    Side effects:
        None.
    Reproducibility implications:
        Keeps clean-checkout failures explicit without inventing scientific
        results or weakening artifact requirements.
    """

    def __init__(
        self,
        *,
        artifact_name: str,
        expected_path: Path,
        producing_stage: str,
        canonical_command: str,
    ) -> None:
        self.artifact_name = artifact_name
        self.expected_path = expected_path
        self.producing_stage = producing_stage
        self.canonical_command = canonical_command
        super().__init__(
            "Required artifact missing: "
            f"{artifact_name}; expected_path={expected_path}; "
            f"producing_stage={producing_stage}; "
            f"canonical_command={canonical_command}"
        )
