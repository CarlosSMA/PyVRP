from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Collection, Tuple

from deap import base, creator, tools

from pyvrp.ProgressPrinter import ProgressPrinter
from pyvrp.Result import Result
from pyvrp.Statistics import Statistics

if TYPE_CHECKING:
    from pyvrp.PenaltyManager import PenaltyManager
    from pyvrp.Population import Population
    from pyvrp._pyvrp import (
        CostEvaluator,
        ProblemData,
        RandomNumberGenerator,
        Solution,
    )
    from pyvrp.search.SearchMethod import SearchMethod
    from pyvrp.stop.StoppingCriterion import StoppingCriterion


class Populacao_MultiObjetivo:
    def __init__(self, solucao: Solution, objetivos: Tuple[float, ...]):
        self.solucao = solucao
        FitnessMulti = getattr(creator, "FitnessMulti")
        self.fitness = FitnessMulti(values=objetivos)


@dataclass
class GeneticAlgorithmParams:
    """
    Parameters for the genetic algorithm.

    Parameters
    ----------
    repair_probability
        Probability (in :math:`[0, 1]`) of repairing an infeasible solution.
        If the reparation makes the solution feasible, it is also added to
        the population in the same iteration.
    num_iters_no_improvement
        Number of iterations without any improvement needed before a restart
        occurs.

    Attributes
    ----------
    repair_probability
        Probability of repairing an infeasible solution.
    num_iters_no_improvement
        Number of iterations without improvement before a restart occurs.

    Raises
    ------
    ValueError
        When ``repair_probability`` is not in :math:`[0, 1]`, or
        ``num_iters_no_improvement`` is negative.
    """

    repair_probability: float = 0.80
    num_iters_no_improvement: int = 20_000

    def __post_init__(self):
        if not 0 <= self.repair_probability <= 1:
            raise ValueError("repair_probability must be in [0, 1].")

        if self.num_iters_no_improvement < 0:
            raise ValueError("num_iters_no_improvement < 0 not understood.")


class GeneticAlgorithm:
    """
    Creates a GeneticAlgorithm instance.

    Parameters
    ----------
    data
        Data object describing the problem to be solved.
    penalty_manager
        Penalty manager to use.
    rng
        Random number generator.
    population
        Population to use.
    search_method
        Search method to use.
    crossover_op
        Crossover operator to use for generating offspring.
    initial_solutions
        Initial solutions to use to initialise the population.
    params
        Genetic algorithm parameters. If not provided, a default will be used.

    Raises
    ------
    ValueError
        When the population is empty.
    """

    def __init__(
        self,
        data: ProblemData,
        penalty_manager: PenaltyManager,
        rng: RandomNumberGenerator,
        population: Population,
        search_method: SearchMethod,
        crossover_op: Callable[
            [
                tuple[Solution, Solution],
                ProblemData,
                CostEvaluator,
                RandomNumberGenerator,
            ],
            Solution,
        ],
        initial_solutions: Collection[Solution],
        params: GeneticAlgorithmParams = GeneticAlgorithmParams(),
    ):
        if len(initial_solutions) == 0:
            raise ValueError("Expected at least one initial solution.")

        self._data = data
        self._pm = penalty_manager
        self._rng = rng
        self._pop = population
        self._search = search_method
        self._crossover = crossover_op
        self._initial_solutions = initial_solutions
        self._params = params

        self._pareto_front = tools.ParetoFront()

        for sol in initial_solutions:
            self._update_pareto_front(sol)

    @property
    def _cost_evaluator(self) -> CostEvaluator:
        return self._pm.cost_evaluator()

    def _get_objectives(self, sol: Solution) -> Tuple[float, ...]:
        num_vehicles = sol.num_routes()
        distance = sol.distance()
        return float(num_vehicles), float(distance)

    def _update_pareto_front(self, sol: Solution) -> bool:
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        if not sol.is_feasible():
            return False

        objectives = self._get_objectives(sol)
        moo_sol = Populacao_MultiObjetivo(sol, objectives)

        front_before = set(self._pareto_front.items)
        self._pareto_front.update([moo_sol])
        front_after = set(self._pareto_front.items)

        return front_before != front_after

    def run(
        self,
        stop: StoppingCriterion,
        collect_stats: bool = True,
        display: bool = False,
        display_interval: float = 5.0,
    ):
        """
        Runs the genetic algorithm with the provided stopping criterion.

        Parameters
        ----------
        stop
            Stopping criterion to use. The algorithm runs until the first time
            the stopping criterion returns ``True``.
        collect_stats
            Whether to collect statistics about the solver's progress. Default
            ``True``.
        display
            Whether to display information about the solver progress. Default
            ``False``. Progress information is only available when
            ``collect_stats`` is also set.
        display_interval
            Time (in seconds) between iteration logs. Defaults to 5s.

        Returns
        -------
        Result
            A Result object, containing statistics (if collected) and the best
            found solution.
        """
        print_progress = ProgressPrinter(display, display_interval)
        print_progress.start(self._data)

        start = time.perf_counter()
        stats = Statistics(collect_stats=collect_stats)
        iters = 0
        iters_no_improvement = 1

        for sol in self._initial_solutions:
            self._pop.add(sol, self._cost_evaluator)

        while not stop(len(self._pareto_front)):
            iters += 1

            if iters_no_improvement == self._params.num_iters_no_improvement:
                print_progress.restart()
                iters_no_improvement = 1
                self._pop.clear()

                for sol in self._initial_solutions:
                    self._pop.add(sol, self._cost_evaluator)

            current_pop = [
                Populacao_MultiObjetivo(pop_sol, self._get_objectives(pop_sol))
                for pop_sol in self._pop
            ]

            if len(current_pop) >= 4:
                current_pop = tools.selNSGA2(current_pop, len(current_pop))

                def tournament(ind1, ind2):
                    if ind1.fitness.dominates(ind2.fitness):
                        return ind1
                    if ind2.fitness.dominates(ind1.fitness):
                        return ind2

                    c_dist1 = getattr(ind1.fitness, "crowding_dist", 0)
                    c_dist2 = getattr(ind2.fitness, "crowding_dist", 0)

                    if c_dist1 > c_dist2:
                        return ind1
                    if c_dist1 < c_dist2:
                        return ind2

                    return ind1 if random.random() < 0.5 else ind2

                cand1, cand2 = random.sample(current_pop, 2)
                cand3, cand4 = random.sample(current_pop, 2)

                parent1 = tournament(cand1, cand2)
                parent2 = tournament(cand3, cand4)

                parents = (parent1.solucao, parent2.solucao)
            else:
                parents = self._pop.select(self._rng, self._cost_evaluator)

            offspring = self._crossover(
                parents, self._data, self._cost_evaluator, self._rng
            )

            improved = self._improve_offspring(offspring)

            if improved:
                iters_no_improvement = 1
            else:
                iters_no_improvement += 1

            stats.collect_from(self._pop, self._cost_evaluator)
            print_progress.iteration(stats)

        end = time.perf_counter() - start

        best_solutions = [
            wrapped.solucao for wrapped in self._pareto_front.items
        ]
        res = Result(best_solutions, stats, iters, end)

        return res

    def _improve_offspring(self, sol: Solution) -> bool:
        improved_front = False

        if self._rng.rand() < 0.50:
            sol = self._search(sol, self._cost_evaluator)

        self._pop.add(sol, self._cost_evaluator)
        self._pm.register(sol)

        if self._update_pareto_front(sol):
            improved_front = True

        # Possibly repair if current solution is infeasible. In that case, we
        # penalise infeasibility more using a penalty booster.
        if (
            not sol.is_feasible()
            and self._rng.rand() < self._params.repair_probability
        ):
            sol = self._search(sol, self._pm.booster_cost_evaluator())

            if sol.is_feasible():
                self._pop.add(sol, self._cost_evaluator)
                self._pm.register(sol)

            if self._update_pareto_front(sol):
                improved_front = True

        return improved_front
