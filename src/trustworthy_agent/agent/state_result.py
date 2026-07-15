"""Structured result returned by a state strategy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from trustworthy_agent.exceptions import StrategyExecutionError

FORBIDDEN_TRANSITION_FACT_KEYS = frozenset(
    {
        "next_state",
        "selected_state",
        "transition",
        "transition_decision",
        "forced_state",
    }
)


@dataclass(frozen=True)
class StateResult:
    """Facts emitted by a state strategy without performing a transition.

    Purpose:
        Carry state-local outputs while preserving the invariant that strategies
        do not select the next FSM state.
    Parameters:
        facts: Structured facts produced inside the current state.
        reason_codes: Machine-readable reasons produced by the strategy.
        provenance: Strategy/config/input identity for derived facts.
    Return value:
        Immutable state result.
    Raised exceptions:
        None.
    Scientific assumptions:
        Facts must be interpreted according to their producing strategy and
        provenance; this class does not validate scientific meaning.
    Side effects:
        None.
    Reproducibility implications:
        Immutable mappings make downstream audit evidence less error-prone.
    """

    facts: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        forbidden = FORBIDDEN_TRANSITION_FACT_KEYS.intersection(self.facts)
        if forbidden:
            raise StrategyExecutionError(
                "State strategies must not encode FSM transitions in StateResult facts: "
                f"{sorted(forbidden)}."
            )
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
