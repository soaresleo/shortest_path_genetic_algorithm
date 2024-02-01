from app.genetic_algorithm import GeneticAlgorithm
from app.instance_generator import StationSet
import pandas as pd

def run_genetic_algorithm(graph_size, seed, generations, pop_size, cross_ratio,
                          mut_ratio, penalty_scalar, filename):
    """
    Run the Genetic Algorithm with the specified parameters.

    Parameters:
    - graph_size (int): Number of nodes in the graph.
    - seed (int or None): Seed for reproducing the same graph. Use an integer or None for a random graph.
    - generations (int): Number of generations for the Genetic Algorithm.
    - pop_size (int): Size of the population in each generation.
    - cross_ratio (float): Crossover probability (0.0 to 1.0).
    - mut_ratio (float): Mutation probability (0.0 to 1.0).
    - penalty_scalar (float): Scalar value for penalty calculation.

    Note: Set seed to None for random graph generation.
    """
    # Generate the graph instance
    graph = StationSet(n=graph_size, s=seed)

    # Initialize the Genetic Algorithm
    GA = GeneticAlgorithm(graph=graph, generations=generations,
                          pop_size=pop_size, cross_ratio=cross_ratio,
                          mut_ratio=mut_ratio, penalty_scalar=penalty_scalar)

    # Run the Genetic Algorithm
    results = GA.evolution()

    # Save the results to a CSV file
    save_ga_results(results[0], results[1], results[2], results[3], filename)

    # Plot the generated graph (optional, comment out if not needed)
    graph.plot_graph()

def save_ga_results(gens_list, objs_list, feas_list, paths_list, filename):
    """
    Save the results of the Genetic Algorithm to a CSV file.

    Parameters:
    - gens_list (list): List of generation IDs.
    - objs_list (list): List of objective values.
    - feas_list (list): List indicating whether the solution is feasible (True/False).
    - paths_list (list): List of paths representing solutions.
    - filename (str): The name of the CSV file to save the results.
    """
    data = {'generation_id': gens_list, 'objective': objs_list,
            'is_feasible': feas_list, 'path': paths_list}
    pd.DataFrame(data).to_csv(filename, index=False)