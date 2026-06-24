import matplotlib.pyplot as plt

import pyvrp
import pyvrp.plotting
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
# The round_func is standard practice in PyVRP to handle VRP benchmark formats correctly
data = pyvrp.read(instance_path, round_func="dimacs")
rng = pyvrp.RandomNumberGenerator(seed=73)

# 2. Set up the C++ components
# Native PyVRP uses PenaltyManager without needing to initialize from data directly
pm = PenaltyManager()

pop_params = PopulationParams(min_pop_size=50, generation_size=100)
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

# 4. Initialize Standard Native Genetic Algorithm
algo = pyvrp.GeneticAlgorithm(
    data=data,
    penalty_manager=pm,
    rng=rng,
    population=pop,
    search_method=ls,
    crossover_op=srex,
    initial_solutions=initial_solutions,
)

# 5. Run the algorithm for 100 generations
print("Running Single-Objective Native PyVRP...")
stop_criterion = MaxIterations(100)

# Run the algorithm natively
result = algo.run(stop_criterion)

# 6. Output the built-in summary statistics
print("\nOptimization Complete!")
print(result)  # Natively prints distance, vehicles, and execution time

# 7. Native Visualization
# PyVRP has built-in tools to plot the convergence and the actual routes found
fig = plt.figure(figsize=(15, 9))

# Plot the algorithm's statistical performance natively
pyvrp.plotting.plot_result(result, data, fig=fig)

plt.tight_layout()
plt.show()
