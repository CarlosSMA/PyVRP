import matplotlib.pyplot as plt

import pyvrp
from pyvrp import PenaltyManager, Population, PopulationParams
from pyvrp.crossover import selective_route_exchange
from pyvrp.diversity import broken_pairs_distance as bpd
from pyvrp.search import (
    NODE_OPERATORS,
    ROUTE_OPERATORS,
    LocalSearch,
    compute_neighbours,
)
from pyvrp.stop import MaxIterations


def get_genetic_algorithm_params():
    data = pyvrp.read("./tests/data/RC208.vrp")
    rng = pyvrp.RandomNumberGenerator(seed=73)

    pm = PenaltyManager.init_from(data)

    pop_params = PopulationParams(min_pop_size=100, generation_size=300)
    pop = Population(bpd, params=pop_params)

    neighbours = compute_neighbours(data)
    ls = LocalSearch(data, rng, neighbours)

    for op in NODE_OPERATORS:
        ls.add_node_operator(op(data))

    for op in ROUTE_OPERATORS:
        ls.add_route_operator(op(data))

    initial_solutions = [
        pyvrp.Solution.make_random(data, rng)
        for _ in range(pop_params.min_pop_size)
    ]

    return data, pm, rng, pop, ls, initial_solutions

def main():
    data, pm, rng, pop, ls, initial_solutions = get_genetic_algorithm_params()

    algo = pyvrp.GeneticAlgorithm(
        data=data,
        penalty_manager=pm,
        rng=rng,
        population=pop,
        search_method=ls,
        crossover_op=selective_route_exchange,
        initial_solutions=initial_solutions
    )

    stop_criterion = MaxIterations(300)

    result = algo.run(stop_criterion)

    criar_grafico(result)

def criar_grafico(result):
    pareto_points = result.cost()

    pareto_points.sort(key=lambda x: (x[0], x[1]))

    vehicles = [p[0] for p in pareto_points]
    distances = [p[1] for p in pareto_points]

    plt.figure(figsize=(8, 6))

    plt.plot(
        vehicles, distances, color="gray", linestyle="--", zorder=1, alpha=0.6
    )

    plt.scatter(
        vehicles,
        distances,
        color="blue",
        s=100,
        zorder=2,
        label="Non-Dominated Solutions",
    )

    for v, d in zip(vehicles, distances):
        plt.annotate(
            f"{d:.1f}",
            (v, d),
            textcoords="offset points",
            xytext=(10, 5),
            ha="left",
        )

    plt.title("Veículos X distância percorrida", fontsize=14, pad=15)
    plt.xlabel("Quantidade de veículos", fontsize=12)
    plt.ylabel("Distância percorrida", fontsize=12)

    plt.xticks(range(min(vehicles), max(vehicles) + 1))

    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()