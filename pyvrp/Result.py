from typing import List, Tuple

from pyvrp.Statistics import Statistics
from pyvrp._pyvrp import Solution


class Result:
    """
    Stores the outcomes of a multi-objective routing optimization run.

    Parameters
    ----------
    best
        The Pareto front containing the best non-dominated solutions found.
    stats
        Statistics collected during the algorithm's execution.
    num_iterations
        Total number of iterations performed.
    runtime
        Total runtime of the algorithm in seconds.
    """

    def __init__(
        self,
        best: List[Solution],
        stats: Statistics,
        num_iterations: int,
        runtime: float,
    ):
        self._best = best
        self._stats = stats
        self._num_iterations = num_iterations
        self._runtime = runtime

    def cost(self) -> List[Tuple[int, float]]:
        """
        MOO REFACTOR: Single-objective cost is no longer mathematically valid.
        This now returns the objective values (Number of Vehicles, Distance)
        for all non-dominated solutions in the Pareto front.
        """
        return [(sol.num_routes(), sol.distance()) for sol in self._best]

    def is_feasible(self) -> bool:
        """
        Returns whether all solutions in the current Pareto front are feasible.
        """
        return all(sol.is_feasible() for sol in self._best)

    def has_statistics(self) -> bool:
        """
        Returns whether detailed statistics were collected during the algorithm's
        execution.
        """
        return self._stats is not None and self._stats.num_iterations > 0

    @property
    def best(self) -> List[Solution]:
        """
        MOO REFACTOR: Returns the set of non-dominated solutions (Pareto front)
        found by the algorithm, rather than a single solution.
        """
        return self._best

    @property
    def stats(self) -> Statistics:
        """
        Returns the runtime statistics object.
        """
        return self._stats

    @property
    def num_iterations(self) -> int:
        """
        Returns the number of iterations performed.
        """
        return self._num_iterations

    @property
    def runtime(self) -> float:
        """
        Returns the algorithm's runtime in seconds.
        """
        return self._runtime

    def summary(self) -> str:
        """
        MOO REFACTOR: Redesigned summary to print a table of the Pareto Front
        trade-offs, filtering out solutions that have identical objective scores.
        """
        # Filter out objective-space duplicates
        unique_objectives = {}
        for sol in self._best:
            vehicles = sol.num_routes()
            distance = round(sol.distance(), 2)

            # Use the tuple (vehicles, distance) as a key to ensure uniqueness
            if (vehicles, distance) not in unique_objectives:
                unique_objectives[(vehicles, distance)] = sol

        lines = [
            "Multi-Objective Result Summary",
            "==============================",
            f"Pareto front size : {len(self._best)} (Total)",
            f"Unique trade-offs : {len(unique_objectives)}",
            f"All feasible      : {self.is_feasible()}",
            f"Iterations        : {self._num_iterations}",
            f"Runtime           : {self._runtime:.3f}s",
            "",
            "Unique Pareto Front Details:",
            "---------------------------------",
            "Sol # | Vehicles | Distance",
            "---------------------------------",
        ]

        # Sort by vehicles, then distance
        sorted_unique = sorted(
            unique_objectives.keys(), key=lambda x: (x[0], x[1])
        )

        for idx, (vehicles, distance) in enumerate(sorted_unique, start=1):
            lines.append(f"{idx:<5} | {vehicles:<8} | {distance:.2f}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
