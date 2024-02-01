import app.runner as r

"""
Run the Genetic Algorithm with the specified parameters.

Parameters:
- graph_size (int): Number of nodes in the graph.
- seed (int or None): Seed for reproducing the same graph. In tests, it was used
  a seed = 30. Set to None for random graphs.
- generations (int): Number of generations for the Genetic Algorithm.
- pop_size (int): Size of the population in each generation.
- cross_ratio (float): Crossover probability.
- mut_ratio (float): Mutation probability.
- penalty_scalar (float): Scalar value for penalty calculation.
- filename (string): Name of the file in which the results will be saved.
"""

if __name__ == '__main__':
    # Set your desired parameters here or use the default values
    r.run_genetic_algorithm(
        graph_size=211,
        seed=30,
        generations=2000,
        pop_size=40,
        cross_ratio=0.4,
        mut_ratio=0.2,
        penalty_scalar=100,
        filename='GA_211_40_2000.csv'
    )
