"""Backward-compatible AgentContext subtype carrying persisted evidence."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.evidence.diagnostic import (
    AcquisitionEvidence,
    PersistedTrendEvidence,
    RiskEvidence,
    SafetyEvidence,
    WindowEvidence,
)


@dataclass(frozen=True)
class EvidenceAgentContext(AgentContext):
    """Extend stable AgentContext with optional persisted evidence fields.

    Parameters
    ----------
    window_evidence : tuple of WindowEvidence
        Persisted per-window classifier evidence.
    trend_evidence : PersistedTrendEvidence or None
        Immutable persisted V6 trend envelope.
    acquisition_evidence : AcquisitionEvidence or None
        Label-free acquisition aggregate.
    risk_evidence : RiskEvidence or None
        Decision-free normalized risk components.
    safety_evidence : SafetyEvidence or None
        Projection to fields consumed by the existing SafetyGuard.
    explanation_references : tuple of str
        Persisted explanation artifact references.
    representation_metadata, model_metadata, healthy_reference_metadata : dict
        Hash and fit-scope metadata required for audit and replay.

    Notes
    -----
    This frozen subtype preserves every stable ``AgentContext`` field and
    method. The stable module remains byte-identical to its V6 protected
    snapshot, while consumers accepting ``AgentContext`` also accept this
    subtype.
    """

    window_evidence: tuple[WindowEvidence, ...] = ()
    trend_evidence: PersistedTrendEvidence | None = None
    acquisition_evidence: AcquisitionEvidence | None = None
    risk_evidence: RiskEvidence | None = None
    safety_evidence: SafetyEvidence | None = None
    explanation_references: tuple[str, ...] = ()
    representation_metadata: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    healthy_reference_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context(cls, context: AgentContext, **evidence_fields: Any) -> EvidenceAgentContext:
        """Copy a stable context and add validated evidence fields.

        Parameters
        ----------
        context : AgentContext
            Stable context whose complete field values must be preserved.
        **evidence_fields : Any
            Evidence subtype fields and explicit stable scalar projections.

        Returns
        -------
        extended : EvidenceAgentContext
            Frozen subtype accepted wherever ``AgentContext`` is expected.

        Raises
        ------
        TypeError
            If an unknown field is supplied.
        """

        stable_values = {item.name: getattr(context, item.name) for item in fields(AgentContext)}
        stable_values.update(evidence_fields)
        return cls(**stable_values)
