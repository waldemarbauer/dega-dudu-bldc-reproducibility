"""Semantic state identifiers for the canonical diagnostic FSM."""

from __future__ import annotations

from enum import StrEnum


class StateId(StrEnum):
    """Stable semantic state identifiers.

    Purpose:
        Provide state identities that code may depend on instead of ordinal
        labels such as `S3`.
    Parameters:
        Enum value supplied by Python enum machinery.
    Return value:
        A semantic state identifier.
    Raised exceptions:
        `ValueError` if constructed from an unknown value.
    Scientific assumptions:
        None; this is workflow metadata, not diagnostic evidence.
    Side effects:
        None.
    Reproducibility implications:
        Stable values are safe to persist in configs, audit logs, and replay
        evidence.
    """

    DATA_ACQUISITION = "DATA_ACQUISITION"
    DATA_VALIDATION = "DATA_VALIDATION"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
    SPLINE_MODELLING = "SPLINE_MODELLING"
    DIAGNOSTIC_INFERENCE = "DIAGNOSTIC_INFERENCE"
    EXPLANATION_GENERATION = "EXPLANATION_GENERATION"
    DECISION_CHECK = "DECISION_CHECK"
    RECOMMENDATION = "RECOMMENDATION"
    ESCALATION = "ESCALATION"
    NO_DECISION = "NO_DECISION"
    AUDIT = "AUDIT"


CANONICAL_SEMANTIC_TO_ORDINAL: dict[StateId, str] = {
    StateId.DATA_ACQUISITION: "S0",
    StateId.DATA_VALIDATION: "S1",
    StateId.FEATURE_EXTRACTION: "S2",
    StateId.SPLINE_MODELLING: "S3",
    StateId.DIAGNOSTIC_INFERENCE: "S4",
    StateId.EXPLANATION_GENERATION: "S5",
    StateId.DECISION_CHECK: "S6",
    StateId.RECOMMENDATION: "S7",
    StateId.ESCALATION: "S8",
    StateId.NO_DECISION: "S9",
    StateId.AUDIT: "S10",
}
