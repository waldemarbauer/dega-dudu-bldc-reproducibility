"""Deterministic RNG adapters for transition policies."""

from __future__ import annotations

import random as random_module
from dataclasses import dataclass, field


@dataclass
class DeterministicRandomGenerator:
    """Seeded pseudo-random stream for replayable transition selection.

    Purpose:
        Provide the minimal RNG contract required by transition policies without
        relying on hidden global random state.
    Parameters:
        seed: Explicit integer seed recorded for replay.
        stream_name: Human-readable stream namespace for audit evidence.
    Return value:
        Deterministic RNG adapter.
    Raised exceptions:
        None during normal construction.
    Scientific assumptions:
        Randomness controls workflow sampling only; it is not diagnostic
        uncertainty or physical process evidence.
    Side effects:
        Mutates only this instance's private pseudo-random stream.
    Reproducibility implications:
        The `stream_id` records the exact seed namespace used by a transition.
    """

    seed: int
    stream_name: str = "transition"
    _rng: random_module.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random_module.Random(self.seed)

    @property
    def stream_id(self) -> str:
        """Stable seed/stream identifier for transition audit evidence."""

        return f"{self.stream_name}:{self.seed}"

    def random(self) -> float:
        """Return the next pseudo-random value in `[0.0, 1.0)`.

        Purpose:
            Satisfy the injected RNG contract for stochastic policies.
        Parameters:
            None.
        Return value:
            Floating-point value in `[0.0, 1.0)`.
        Raised exceptions:
            None.
        Scientific assumptions:
            None.
        Side effects:
            Advances this instance's private RNG stream.
        Reproducibility implications:
            Same seed and call order reproduce the same sequence.
        """

        return self._rng.random()
