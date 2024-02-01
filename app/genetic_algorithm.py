import numpy as np
import app.instance_generator as ig
import app.path_grower as pg
import math
import networkx as nx

class GeneticAlgorithm:
    """
    Genetic Algorithm for solving optimization problems on a graph.

    Attributes:
    - __graph: StationSet object, representing the problem instance.
    - __generations: Number of generations for the genetic algorithm.
    - __pop_size: Size of the population in each generation.
    - __cross_ratio: Crossover probability.
    - __mut_ratio: Mutation probability.
    - __penalty_scalar: Scalar value for penalty calculation.
    - __population: List of individuals in the population.
    - __paths_list: List of paths corresponding to the individuals.
    - __objectives: List of fitness values of the population.
    - __feasibilities: List of feasibility status for each individual.
    - __best_obj_index: Index of the best individual in the population.
    - __best_individual: Best individual in the population.
    - __best_path: Path corresponding to the best individual.
    - __best_objective: Fitness value of the best individual.
    - __best_feasibility: Feasibility status of the best individual.
    """

    def __init__(self, graph, generations, pop_size, cross_ratio, mut_ratio,
                 penalty_scalar):
        """
        Initializes the GeneticAlgorithm class.

        Parameters:
        - graph: StationSet object, representing the problem instance.
        - generations: Number of generations for the genetic algorithm.
        - pop_size: Size of the population in each generation.
        - cross_ratio: Crossover probability.
        - mut_ratio: Mutation probability.
        - penalty_scalar: Scalar value for penalty calculation.
        """
        self.__graph = graph
        self.__generations = generations
        self.__pop_size = pop_size
        self.__cross_ratio = cross_ratio
        self.__mut_ratio = mut_ratio
        self.__penalty_scalar = penalty_scalar
        self.__population = []
        self.__paths_list = []
        self.__objectives = []
        self.__feasibilities = []
        self.__best_obj_index = None
        self.__best_individual = None
        self.__best_path = None
        self.__best_objective = None
        self.__best_feasibility = None
        self.__initialize_population()

    def __initialize_population(self):
        """
        Generates the initial population.
        """
        while len(self.__population) < self.__pop_size:
            self.__generate_individual()

    def __generate_individual(self):
        """
        Generates a new individual and adds it to the population if it is unique.
        """
        path_data = pg.raise_path(self.__graph.adj_matrix,
                                  len(self.__graph.adj_matrix) - 1)
        individual, path, has_dead_node = \
            path_data[0], path_data[1],path_data[2]

        # Check uniqueness of the individual and path
        if any(np.array_equal(individual, x) for x in self.__population) or \
           any(np.array_equal(path, x) for x in self.__paths_list):
            return

        fitness = self.__calculate_fitness(path, has_dead_node)
        self.__population.append(individual)
        self.__paths_list.append(np.array(self.__graph.G.nodes)[path])
        self.__objectives.append(fitness)
        self.__feasibilities.append(not has_dead_node)

        self.__update_best_individual()

    def __calculate_fitness(self, path, has_dead_node):
        """
        Calculates the fitness value of an individual path.

        Parameters:
        - path: List of nodes representing the path.
        - has_dead_node: Boolean indicating if the path has a dead node.

        Returns:
        - objective: Fitness value of the path.
        """
        graph_nodes = np.array(self.__graph.G.nodes)
        path_nodes = graph_nodes[path]
        weight, penalty = 0, 0

        if has_dead_node:
            last_node, reduced_path_len, degree = -2, len(path) - 1, \
                len(self.__graph.G.nodes)
            penalty_power = 1 / (reduced_path_len / degree)
            penalty = self.__penalty_scalar ** penalty_power
        else:
            last_node = -1

        for node in range(len(path_nodes) + last_node):
            i, j = graph_nodes[path[node]], graph_nodes[path[node + 1]]
            weight += self.__graph.G.get_edge_data(i, j)['weight']

        objective = weight + (penalty * has_dead_node)
        return objective

    def __update_best_individual(self):
        """
        Updates the information about the best individual in the population.
        """
        current_best_obj_index = np.argmin(self.__objectives)
        if self.__best_obj_index is None \
            or self.__objectives[current_best_obj_index] < \
                self.__objectives[self.__best_obj_index]:
            self.__best_obj_index = current_best_obj_index
            self.__best_individual = self.__population[current_best_obj_index]
            self.__best_path = self.__paths_list[current_best_obj_index]
            self.__best_objective = self.__objectives[current_best_obj_index]
            self.__best_feasibility = \
                self.__feasibilities[current_best_obj_index]

    def spin_roulette(self):
        """
        Selects two parents using roulette wheel selection.

        Returns:
        - couple: Tuple containing two selected parents.
        """
        inverse_objectives = [1 / obj for obj in self.__objectives]
        probs = [obj / np.sum(inverse_objectives) for obj in inverse_objectives]
        parents_indexes = np.random.choice(a=len(probs), size=2,
                                           replace=False, p=probs)
        parents = np.array(self.__population)[parents_indexes]
        major_parent_index = \
            np.argmin(np.array(self.__objectives)[parents_indexes])
        minor_parent_index = 1 - major_parent_index
        major_parent, minor_parent = parents[major_parent_index], \
            parents[minor_parent_index]
        couple = (major_parent, minor_parent)
        return couple

    def make_crossover(self, parent_1, parent_2):
        """
        Performs crossover to create a child from two parents.

        Parameters:
        - parent_1: First parent individual.
        - parent_2: Second parent individual.

        Returns:
        - child: Offspring generated through crossover.
        """
        parent_1_genes_qty = int(math.ceil(len(parent_1)) / 2)
        parent_1_chosen_genes = np.random.choice(a=parent_1,
                                                 size=parent_1_genes_qty,
                                                 replace=False)
        child = np.where(np.isin(parent_1, parent_1_chosen_genes),
                         parent_1, None)
        fill_in = np.where(child == None)[0]
        i_ = 0
        for element in parent_2:
            if element in child:
                continue
            else:
                child[fill_in[i_]] = element
                i_ += 1
        return child

    def mutate(self, individual):
        """
        Mutates an individual by swapping two randomly chosen genes.

        Parameters:
        - individual: Individual to be mutated.

        Returns:
        - mutated_individual: Mutated individual.
        """
        chosen_genes = np.random.choice(individual, size=2, replace=False)
        genes_indexes = np.where(np.isin(individual, chosen_genes))[0]
        gene_a, gene_b = \
            individual[genes_indexes[0]], individual[genes_indexes[1]]
        individual[genes_indexes[0]], individual[genes_indexes[1]] = \
            gene_b, gene_a
        return individual

    def reproduce(self):
        """
        Performs reproduction to generate a new individual.

        Returns:
        - child: Newly generated individual.
        """
        couple = self.spin_roulette()
        u = np.random.uniform()
        if u <= self.__cross_ratio:
            child = self.make_crossover(couple[0], couple[1])
            v = np.random.uniform()
            if v <= self.__mut_ratio:
                child = self.mutate(child)
            return child
        else:
            child = couple[np.random.randint(0, 2)]
            v = np.random.uniform()
            if v <= self.__mut_ratio:
                child = self.mutate(child)
            return child

    def raise_population(self):
        """
        Generates a new population for the next generation.
        """
        new_population, new_paths_list, new_objectives, new_feasibilities = \
            [], [], [], []
        while len(new_population) < self.__pop_size:
            offspring = self.reproduce()
            path_data = pg.raise_path(self.__graph.adj_matrix,
                                      len(self.__graph.adj_matrix) - 1,
                                      priorities=offspring)
            path = path_data[1]

            # Check uniqueness of the offspring and path
            if any(np.array_equal(offspring, x) for x in new_population) or \
               any(np.array_equal(path, x) for x in new_paths_list):
                continue

            has_dead_node = path_data[2]
            fitness = self.__calculate_fitness(path, has_dead_node)
            new_population.append(offspring)
            new_paths_list.append(np.array(self.__graph.G.nodes)[path])
            new_objectives.append(fitness)
            new_feasibilities.append(not has_dead_node)

        # Replace a random individual with the best individual if not present
        if not any(np.array_equal(self.__best_individual, x) \
                   for x in new_population):
            choice = np.random.randint(0, self.__pop_size)
            new_population[choice] = self.__best_individual
            new_paths_list[choice] = self.__best_path
            new_objectives[choice] = self.__best_objective
            new_feasibilities[choice] = self.__best_feasibility

        # Update the population with the new individuals
        self.__population = new_population
        self.__paths_list = new_paths_list
        self.__objectives = new_objectives
        self.__feasibilities = new_feasibilities
        
        self.__update_best_individual()

    def evolution(self):
        """
        Executes the evolution process for the given number of generations.

        Returns:
        - gens_list (list): List of generation IDs.
        - bests_objs_list (list): List of best objective values.
        - bests_feasibs_list (list): List indicating feasibility of solutions (True/False).
        - bests_paths_list (list): List of paths representing solutions.
        """
        gens_list = []
        bests_objs_list = []
        bests_feasibs_list = []
        bests_paths_list = []

        # Run evolution for the specified number of generations
        for i in range(self.__generations):
            self.raise_population()

            # Format generation string for printing
            gen_str_completion = 5 - len(str(i + 1))
            gen = f"Gen {gen_str_completion * '0' + str(i + 1)}"

            # Extract relevant information for the current generation
            b_obj = self.__best_objective
            b_feas = str(self.__best_feasibility)
            b_path = self.__best_path

            # Append information to respective lists
            gens_list.append(gen)
            bests_objs_list.append(b_obj)
            bests_feasibs_list.append(b_feas)
            bests_paths_list.append(b_path)

            # Print information for the current generation
            print(f"{gen} |", "Best obj: {:.6f} |".format(self.__best_objective),
                  f"Feasible: {b_feas} |", f"Path: {b_path}")

        # Calculate and print results using different shortest path algorithms
        last_node = list(self.__graph.G.nodes)[-1]
        methods = ['dijkstra', 'bellman-ford']
        methods_alias = ['Dijkstra ', 'Bell-ford']

        gens_list += methods_alias

        for i, method in enumerate(methods):
            # Calculate shortest path using the specified method
            sp_length = nx.shortest_path_length(self.__graph.G, source="1",
                                                target=last_node,
                                                weight='weight',
                                                method=method)
            sp_nodes = np.array(
                nx.shortest_path(self.__graph.G, source="1", target=last_node,
                                 weight='weight', method=method))

            # Append information to respective lists
            bests_objs_list.append(sp_length)
            bests_feasibs_list.append(str(True))
            bests_paths_list.append(sp_nodes)

            # Print information for the calculated shortest path
            print(f"{methods_alias[i]} |",
                  "Best obj: {:.6f} |".format(sp_length),
                  f"Feasible: {True} |",
                  f"Path: {sp_nodes}")

        return gens_list, bests_objs_list, bests_feasibs_list, bests_paths_list