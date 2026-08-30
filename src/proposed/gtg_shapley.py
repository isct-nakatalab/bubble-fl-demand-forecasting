import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class GTGShapley:
    """
    GTG-Shapley contribution calculator.
    """

    def __init__(
        self,
        players: list[Any],
        metric_fn: Callable[[tuple[Any, ...]], float],
        *,
        initial_metric: float,
        eps: float = 1e-3,
        converge_min: int | None = None,
        last_k: int = 10,
        converge_criteria: float = 0.05,
        max_percentage: float = 0.8,
        seed: int | None = None,
    ):
        self.players = list(players)
        self.metric_fn = metric_fn
        self.initial_metric = float(initial_metric)

        self.eps = float(eps)
        self.converge_min = converge_min or max(30, len(self.players))
        self.last_k = int(last_k)
        self.converge_criteria = float(converge_criteria)
        self.max_percentage = float(max_percentage)

        self.rng = np.random.default_rng(seed)

        n = len(self.players)
        self.max_number = int(
            min(
                2**n,
                max(self.converge_min, self.max_percentage * (2**n)),
            )
        )

        self.metric_cache: dict[tuple[Any, ...], float] = {}

    def _canonical_subset(self, subset: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(sorted(subset))

    def _get_metric(self, subset: tuple[Any, ...]) -> float:
        subset = self._canonical_subset(subset)

        if len(subset) == 0:
            return self.initial_metric

        if subset not in self.metric_cache:
            self.metric_cache[subset] = float(self.metric_fn(subset))

        return self.metric_cache[subset]

    def compute(self) -> dict[Any, float]:
        n = len(self.players)

        if n == 0:
            return {}

        full_subset = self._canonical_subset(self.players)
        full_metric = self._get_metric(full_subset)

        contribution_records: list[np.ndarray] = []
        index = 0

        while self._not_convergent(index, contribution_records):
            for first_player in self.players:
                index += 1

                others = [p for p in self.players if p != first_player]
                perm = [first_player] + list(self.rng.permutation(others))

                values = np.zeros(n + 1)
                values[0] = self.initial_metric

                marginal = np.zeros(n)

                for j in range(n):
                    subset = self._canonical_subset(perm[: j + 1])

                    # GTG truncation:
                    # full_metric にかなり近ければ、それ以降の評価を省略
                    if abs(full_metric - values[j]) >= self.eps:
                        values[j + 1] = self._get_metric(subset)
                    else:
                        values[j + 1] = values[j]

                    added_player = perm[j]
                    added_idx = self.players.index(added_player)
                    marginal[added_idx] = values[j + 1] - values[j]

                contribution_records.append(marginal)

                if not self._not_convergent(index, contribution_records):
                    break

        mean_contrib = np.mean(contribution_records, axis=0)

        return {player: float(mean_contrib[i]) for i, player in enumerate(self.players)}

    def _not_convergent(
        self,
        index: int,
        contribution_records: list[np.ndarray],
    ) -> bool:
        if index >= self.max_number:
            return False

        if index <= self.converge_min:
            return True

        records = np.asarray(contribution_records)

        running_means = np.cumsum(records, axis=0) / np.arange(1, len(records) + 1).reshape(-1, 1)

        recent = running_means[-self.last_k :]
        latest = recent[-1:]

        errors = np.mean(
            np.abs(recent - latest) / (np.abs(latest) + 1e-12),
            axis=1,
        )

        return bool(np.max(errors) > self.converge_criteria)
