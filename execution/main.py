import pyvrp
from pyvrp import PenaltyManager, Population, PopulationParams
from pyvrp.crossover import selective_route_exchange as srex
from pyvrp.diversity import broken_pairs_distance as bpd
from pyvrp.search import (
    NODE_OPERATORS,
    ROUTE_OPERATORS,
    LocalSearch,
    compute_neighbours,
)
from pyvrp.stop import MaxIterations

# 1. Load a standard benchmark instance
# Point this to any valid VRPLIB file in your test directory.
instance_path = "../tests/data/RC208.vrp"
data = pyvrp.read(instance_path)
rng = pyvrp.RandomNumberGenerator(seed=73)

# 2. Set up the C++ components
# FIX: Use the factory method to initialize the 3-dimensional penalty tuple dynamically
pm = PenaltyManager.init_from(data)

pop_params = PopulationParams(min_pop_size=50, generation_size=100)
# pop_params = PopulationParams()
# FIX: Pass the broken_pairs_distance (bpd) diversity metric
pop = Population(bpd, params=pop_params)

# Setup local search with default operators
neighbours = compute_neighbours(data)
ls = LocalSearch(data, rng, neighbours)

for op in NODE_OPERATORS:
    ls.add_node_operator(op(data))

for op in ROUTE_OPERATORS:
    ls.add_route_operator(op(data))

# 3. Generate initial random solutions
initial_solutions = [
    pyvrp.Solution.make_random(data, rng)
    for _ in range(pop_params.min_pop_size)
]

# 4. Initialize YOUR modified Multi-Objective Genetic Algorithm
algo = pyvrp.GeneticAlgorithm(
    data=data,
    penalty_manager=pm,
    rng=rng,
    population=pop,
    search_method=ls,
    crossover_op=srex,
    initial_solutions=initial_solutions,
)

# 5. Run the algorithm for 2500 generations
print("Running Multi-Objective NSGA-II VRP...")
stop_criterion = MaxIterations(2500)

# display=True will trigger your custom ProgressPrinter and Result output
result = algo.run(stop_criterion, display=True)

# 6. Print your new Pareto Front summary table
print("\nOptimization Complete!")
print(result)

import matplotlib.pyplot as plt


def plot_pareto_front(result):
    # 1. Extract the objective values from your custom Result object
    # result.cost() returns [(vehicles, distance), ...]
    pareto_points = result.cost()

    if not pareto_points:
        print("No solutions found to plot.")
        return

    # 2. Sort the points by Number of Vehicles (X) and then Distance (Y)
    # This is required so the line connects the points in the correct order
    pareto_points.sort(key=lambda x: (x[0], x[1]))

    vehicles = [p[0] for p in pareto_points]
    distances = [p[1] for p in pareto_points]

    # 3. Create the plot
    plt.figure(figsize=(8, 6))

    # Draw the Pareto curve (dashed line)
    plt.plot(
        vehicles, distances, color="gray", linestyle="--", zorder=1, alpha=0.6
    )

    # Plot the actual solution points
    plt.scatter(
        vehicles,
        distances,
        color="blue",
        s=100,
        zorder=2,
        label="Non-Dominated Solutions",
    )

    # Annotate each point with its exact distance
    for v, d in zip(vehicles, distances):
        plt.annotate(
            f"{d:.1f}",
            (v, d),
            textcoords="offset points",
            xytext=(10, 5),
            ha="left",
        )

    # 4. Formatting
    plt.title(
        "Pareto Front: Fleet Size vs. Total Distance", fontsize=14, pad=15
    )
    plt.xlabel("Number of Vehicles (Minimize)", fontsize=12)
    plt.ylabel("Total Distance (Minimize)", fontsize=12)

    # Force X-axis to show only whole numbers (you can't have 2.5 vehicles)
    plt.xticks(range(min(vehicles), max(vehicles) + 1))

    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # 5. Render the chart
    plt.show()


# --- Add this right at the very end of your main.py file ---
plot_pareto_front(result)
