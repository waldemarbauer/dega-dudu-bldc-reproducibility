"""Common strategy protocol shared by all S3-S7 strategy slots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.state_result import StateResult


@runtime_checkable
class StateStrategy(Protocol):
    """Protocol for replaceable state strategies.

    Purpose:
        Define the common contract for strategy implementations without coupling
        the FSM engine to concrete S3-S7 classes.
    Parameters:
        Implementations expose `strategy_name` and `strategy_version`.
    Return value:
        Protocol type used for structural checks.
    Raised exceptions:
        Implementations may raise configuration or execution errors.
    Scientific assumptions:
        Strategy-specific assumptions must be documented by implementations.
    Side effects:
        Strategies must not mutate source data, global RNG state, or perform FSM
        transitions directly.
    Reproducibility implications:
        Stable name/version support strategy ablations and audit identity.
    """

    strategy_name: str
    strategy_version: str

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Validate strategy configuration before a run starts."""

    def execute(self, context: AgentContext) -> StateResult:
        """Execute state-local computation and return structured facts."""


class StateStrategyABC(ABC):
    """Abstract base class for nominal state strategy implementations.

    Purpose:
        Offer a conventional inheritance contract for strategies while the
        engine itself remains coupled only to the structural protocol.
    Parameters:
        Implementations provide stable strategy identity and state-local
        execution.
    Return value:
        Abstract base class.
    Raised exceptions:
        TypeError if instantiated without required implementation members.
    Scientific assumptions:
        Strategy subclasses must document their own scientific assumptions.
    Side effects:
        Subclasses must not perform FSM transitions or bypass SafetyGuard.
    Reproducibility implications:
        Name/version identity is required for audit and strategy substitution.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Stable strategy name from configuration."""

    @property
    @abstractmethod
    def strategy_version(self) -> str:
        """Stable strategy implementation version."""

    @abstractmethod
    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Validate strategy configuration before a run starts."""

    @abstractmethod
    def execute(self, context: AgentContext) -> StateResult:
        """Execute state-local computation and return structured facts."""
