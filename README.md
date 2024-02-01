# Genetic Algorithm for the Shortest Path Problem

This project represents my final work for the class Heuristic Search Techniques, lectured by Luciano Ferreira, at the Business master's program in Operations Research, from the Federal University of Rio Grande do Sul (Universidade Federal do Rio Grande do Sul - UFRGS). The implementation is based on the Genetic Algorithm (GA) proposed by Gent, Cheng, and Wang in their 1997 paper ["Genetic Algorithms for Solving Shortest Path Problems"](https://ieeexplore.ieee.org/abstract/document/592343), available on IEEE Xplore.

While the code closely follows the algorithm outlined by the authors, certain modifications were made to accommodate graph generation. Instead of using specific instances as the authors did, I opted for a graph generator inspired by the work of Guedes and Borenstein in their 2015 paper ["Column generation based heuristic framework for the multiple-depot vehicle type scheduling problem"](https://www.sciencedirect.com/science/article/abs/pii/S0360835215003976).

The main experiment involved a graph with 211 nodes, a population size of 40, crossover rate of 0.4, mutation rate of 0.2, and a termination criterion of 2000 generations. These parameters are the same used by Gent, Cheng, and Wang, except by the number of edges. The algorithm's performance was evaluated against the optimal solution obtained through the NetworkX library's shortest path functions.

The focus of this project is not on computational time but on exploring how the shortest path problem can be encoded into chromosomes and solved using a GA. The results show that, with the specified configuration, the algorithm consistently converges to the optimal solution.

## Usage

To run the Genetic Algorithm with your desired parameters, you can use the `main.py` file. Modify the parameters within the script as needed and execute it:

```
# main.py

import app.runner as r

if __name__ == '__main__':
    # Set your desired parameters here or use the default values
    r.run_genetic_algorithm(
        graph_size=20,
        seed=30,
        generations=50,
        pop_size=10,
        cross_ratio=0.4,
        mut_ratio=0.2,
        penalty_scalar=100,
        filename='GA_211_40_2000.csv'
    )
```

## Modules

### genetic_algorithm.py
This module contains the implementation of the Genetic Algorithm for solving optimization problems on a graph. It includes functionalities such as initialization, population generation, crossover, mutation, reproduction, and evolution.

### instance_generator.py
The StationSet class in this module represents a set of stations with methods for generating a graph and plotting it. It includes functionalities to generate an adjacency matrix, create a NetworkX graph, transform positions for plotting, calculate distances between nodes, and add weights to the graph.

### path_grower.py
This module contains functions for growing a path through an adjacency array until reaching the end node. Functions include generating unique random priorities, building a 2D array with ones and a column of zeros, getting eligible nodes at a given time step, adding a node to the path based on priorities and eligible nodes, and raising a path through the adjacency array.

### runner.py
The run_genetic_algorithm function in this module provides a convenient way to execute the Genetic Algorithm with specified parameters. Modify the parameters within this function according to your requirements. The save_ga_results function saves the results to a CSV file.

## References

- GENT, Mitsuo; CHENG, Runwei; WANG, Dingwei. <b>Genetic Algorithms for Solving Shortest Path Problems</b>.  Proceedings of 1997 IEEE International Conference on Evolutionary Computation. Indianapolis, 1997.

- GUEDES, Pablo C.; BORENSTEIN, Denis. <b>Column generation based heuristic framework for the multiple-depot vehicle type scheduling problem</b>. Computers & Industrial Engineering, v.90, p.361-370. 2015.

- TALBI, El-Ghazali. <b>Metaheuristics: From Design to Implementation</b>. John Wiley & Sons. 2009.

