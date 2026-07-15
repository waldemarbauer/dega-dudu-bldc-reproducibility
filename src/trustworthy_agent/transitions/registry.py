"""Transition-policy registry boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from trustworthy_agent.exceptions import TransitionPolicyError
from trustworthy_agent.transitions.base import TransitionPolicy

TransitionPolicyFactory: TypeAlias = type[Any]


@dataclass(frozen=True)
class TransitionPolicyDescriptor:
    """Registered transition-policy identity.

    Purpose:
        Associate a stable policy name/version with an implementation class
        without importing concrete policies in the FSM engine.
    Parameters:
        name: Stable policy name from configuration.
        version: Policy implementation version.
        implementation_path: Import path recorded for provenance.
        implementation: Class satisfying the `TransitionPolicy` protocol.
    Return value:
        Immutable descriptor.
    Raised exceptions:
        None.
    Scientific assumptions:
        Transition probabilities are workflow-control evidence, not diagnostic
        confidence or physical degradation dynamics.
    Side effects:
        None.
    Reproducibility implications:
        Descriptor identity can be stored in manifests and audit events.
    """

    name: str
    version: str
    implementation_path: str
    implementation: TransitionPolicyFactory

    def create(self) -> TransitionPolicy:
        """Instantiate the registered policy implementation.

        Purpose:
            Keep policy construction centralized in the registry boundary.
        Parameters:
            None.
        Return value:
            Transition policy instance.
        Raised exceptions:
            Implementation constructor exceptions propagate.
        Scientific assumptions:
            None.
        Side effects:
            Implementation-specific construction effects, if any.
        Reproducibility implications:
            Exact implementation identity remains attached to the descriptor.
        """

        return cast(TransitionPolicy, self.implementation())


class TransitionPolicyRegistry:
    """Registry for resolving transition policies by stable name/version.

    Purpose:
        Keep transition-policy substitution outside the FSM engine.
    Parameters:
        None.
    Return value:
        Mutable registry instance.
    Raised exceptions:
        TransitionPolicyError for duplicate or unresolved policy identity.
    Scientific assumptions:
        None.
    Side effects:
        Mutates only its private in-memory mapping.
    Reproducibility implications:
        Exact policy resolution supports replay and experiment fingerprints.
    """

    def __init__(self) -> None:
        self._descriptors: dict[tuple[str, str], TransitionPolicyDescriptor] = {}

    def register(self, descriptor: TransitionPolicyDescriptor) -> None:
        """Register one transition-policy descriptor.

        Purpose:
            Add a replaceable transition policy implementation.
        Parameters:
            descriptor: Policy metadata and implementation class.
        Return value:
            None.
        Raised exceptions:
            TransitionPolicyError on duplicate identity or protocol mismatch.
        Scientific assumptions:
            None.
        Side effects:
            Updates the registry mapping.
        Reproducibility implications:
            Rejects ambiguous policy identity.
        """

        key = (descriptor.name, descriptor.version)
        if key in self._descriptors:
            raise TransitionPolicyError(f"Duplicate transition-policy registration: {key}")
        required = ("policy_name", "policy_version", "decide")
        if not all(hasattr(descriptor.implementation, name) for name in required):
            raise TransitionPolicyError("Implementation must satisfy TransitionPolicy.")
        self._descriptors[key] = descriptor

    def resolve(self, policy_name: str, version: str | None = None) -> TransitionPolicyDescriptor:
        """Resolve a transition policy descriptor.

        Purpose:
            Look up policy identity without hardcoding a policy in the engine.
        Parameters:
            policy_name: Stable configured policy name.
            version: Optional exact policy version.
        Return value:
            Matching policy descriptor.
        Raised exceptions:
            TransitionPolicyError if no unique match exists.
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
            if key[0] == policy_name and (version is None or key[1] == version)
        ]
        if len(matches) != 1:
            raise TransitionPolicyError(
                f"Expected one transition policy for name={policy_name} version={version}; "
                f"found {len(matches)}."
            )
        return matches[0]
