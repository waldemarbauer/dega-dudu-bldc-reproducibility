"""Strategy registry boundary for replaceable state strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from trustworthy_agent.agent.states import StateId
from trustworthy_agent.exceptions import StrategyResolutionError
from trustworthy_agent.strategies.base import StateStrategy

StrategyFactory: TypeAlias = type[Any]


@dataclass(frozen=True)
class StrategyDescriptor:
    """Registered strategy identity.

    Purpose:
        Associate a stable strategy name/version with a semantic state without
        importing concrete implementations in the engine.
    Parameters:
        state: Semantic state served by the strategy.
        name: Stable strategy name from configuration.
        version: Strategy implementation version.
        implementation_path: Import path recorded for provenance.
        implementation: Class satisfying the `StateStrategy` protocol.
    Return value:
        Immutable descriptor.
    Raised exceptions:
        None.
    Scientific assumptions:
        None; strategy science belongs to concrete implementations.
    Side effects:
        None.
    Reproducibility implications:
        Descriptor fields can be recorded in manifests and audit events.
    """

    state: StateId
    name: str
    version: str
    implementation_path: str
    implementation: StrategyFactory

    def create(self) -> StateStrategy:
        """Instantiate the registered strategy implementation.

        Purpose:
            Centralize strategy construction while keeping the engine coupled to
            the strategy contract rather than concrete S3-S7 implementations.
        Parameters:
            None.
        Return value:
            State strategy instance.
        Raised exceptions:
            Implementation constructor exceptions propagate.
        Scientific assumptions:
            None.
        Side effects:
            Implementation-specific construction effects, if any.
        Reproducibility implications:
            Exact implementation identity remains attached to the descriptor.
        """

        return cast(StateStrategy, self.implementation())


class StrategyRegistry:
    """Registry for resolving strategies by semantic state and stable name.

    Purpose:
        Keep strategy substitution outside the FSM engine.
    Parameters:
        None.
    Return value:
        Mutable registry instance.
    Raised exceptions:
        StrategyResolutionError for duplicates or missing strategies.
    Scientific assumptions:
        None.
    Side effects:
        Mutates only its private in-memory registry.
    Reproducibility implications:
        Resolution can later be serialized to audit/run manifests.
    """

    def __init__(self) -> None:
        self._descriptors: dict[tuple[StateId, str, str], StrategyDescriptor] = {}

    def register(self, descriptor: StrategyDescriptor) -> None:
        """Register one strategy descriptor.

        Purpose:
            Add a replaceable strategy implementation for a state slot.
        Parameters:
            descriptor: Strategy metadata and implementation class.
        Return value:
            None.
        Raised exceptions:
            StrategyResolutionError on duplicate identity or protocol mismatch.
        Scientific assumptions:
            None.
        Side effects:
            Updates the registry mapping.
        Reproducibility implications:
            Rejects ambiguous strategy identity.
        """

        key = (descriptor.state, descriptor.name, descriptor.version)
        if key in self._descriptors:
            raise StrategyResolutionError(f"Duplicate strategy registration: {key}")
        required = ("strategy_name", "strategy_version", "validate_config", "execute")
        if not all(hasattr(descriptor.implementation, name) for name in required):
            raise StrategyResolutionError("Implementation must satisfy StateStrategy.")
        self._descriptors[key] = descriptor

    def resolve(
        self, state: StateId, strategy_name: str, version: str | None = None
    ) -> StrategyDescriptor:
        """Resolve a strategy descriptor for a semantic state.

        Purpose:
            Look up strategy identity without importing concrete classes in the
            engine.
        Parameters:
            state: Semantic state slot.
            strategy_name: Stable configured strategy name.
            version: Optional exact strategy version.
        Return value:
            Matching strategy descriptor.
        Raised exceptions:
            StrategyResolutionError if no unique match exists.
        Scientific assumptions:
            None.
        Side effects:
            None.
        Reproducibility implications:
            Exact version resolution is deterministic.
        """

        matches = [
            descriptor
            for key, descriptor in self._descriptors.items()
            if key[0] == state
            and key[1] == strategy_name
            and (version is None or key[2] == version)
        ]
        if len(matches) != 1:
            raise StrategyResolutionError(
                f"Expected one strategy for state={state} name={strategy_name} version={version}; "
                f"found {len(matches)}."
            )
        return matches[0]
