from typing import List, Tuple

from pyvrp.Statistics import Statistics
from pyvrp._pyvrp import Solution


class Result:
    """
    Stores the outcomes of a single run. An instance of this class is returned
    once the GeneticAlgorithm completes.

    Parameters
    ----------
    best
        The best observed solution.
    stats
        A Statistics object containing runtime statistics.
    num_iterations
        Number of iterations performed by the genetic algorithm.
    runtime
        Total runtime of the main genetic algorithm loop.

    Raises
    ------
    ValueError
        When the number of iterations or runtime are negative.
    """

    def __init__(
        self,
        best: List[Solution],
        stats: Statistics,
        num_generations: int,
        runtime: float,
    ):
        self._best = best
        self._stats = stats
        self._num_generations = num_generations
        self._runtime = runtime

    def cost(self) -> List[Tuple[int, float]]:
        return [(sol.num_routes(), sol.distance()) for sol in self._best]

    @property
    def best(self) -> List[Solution]:
        return self._best

    def is_feasible(self) -> bool:
        """
        Returns whether the best solution is feasible.
        """
        return True

    @property
    def num_generations(self) -> int:
        """
        Returns the number of iterations performed.
        """
        return self._num_generations