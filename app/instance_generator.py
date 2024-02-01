import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx

class StationSet:
    """
    Represents a set of stations with methods for generating a graph and
    plotting it.

    Attributes:
    - n: Number of stations in the set.
    - allow_replacement: Flag indicating whether replacement is allowed when
      generating adjacency matrix.
    - scalar: Scalar value used for transforming positions.
    - s: Seed for random number generation.
    - adj_matrix: Adjacency matrix representing connections between stations.
    - G: NetworkX graph representing the station set.
    - pos: Dictionary mapping nodes to positions in the graph.
    - transformed_positions: Transformed positions for plotting.
    - distances: Dictionary of distances between nodes.
    """

    def __init__(self, n, allow_replacement=True, scalar=700, s=None):
        """
        Initializes the StationSet instance.

        Parameters:
        - n: Number of stations in the set.
        - allow_replacement: Flag indicating whether replacement is allowed when
          generating adjacency matrix.
        - scalar: Scalar value used for transforming positions.
        - s: Seed for random number generation.
        """
        self.s = s
        self.n = n
        self.allow_replacement = allow_replacement
        self.scalar = scalar
        self.adj_matrix = self.generate_adj_matrix()
        self.G, self.pos = self.generate_graph()
        self.transformed_positions = self.transform_positions()
        self.distances = self.calculate_distances(self.G,
                                                  self.transformed_positions)
        self.G = self.add_weights(self.G)

    def generate_adj_matrix(self):
        """
        Generates an adjacency matrix for the station set.

        Returns:
        - adj_matrix: Generated adjacency matrix.
        """
        adj_matrix = [[0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]]
        memory = []

        while len(adj_matrix) < self.n:
            arcs = list(zip(*np.where(np.array(adj_matrix) == 1)))

            if self.s is not None:
                random.seed(self.s)

            selected_arc = random.choice(arcs)
            selected_arc = (min(selected_arc), max(selected_arc))

            if not self.allow_replacement and selected_arc in memory:
                continue

            m = len(adj_matrix)

            for l in range(m):
                adj_matrix[l].append(1 if l == selected_arc[0] or \
                                     l == selected_arc[1] else 0)

            new_line = [0] * (m + 1)
            new_line[selected_arc[0]] = 1
            new_line[selected_arc[1]] = 1
            adj_matrix.append(new_line)
            memory.append(selected_arc)

        return adj_matrix

    def generate_graph(self):
        """
        Generates a NetworkX graph for the station set.

        Returns:
        - G: Generated NetworkX graph.
        - pos: Dictionary mapping nodes to positions in the graph.
        """
        G = nx.Graph()

        # Add nodes to the graph with labels from "A" to the ith alphabet letter
        for i in range(len(self.adj_matrix)):
            node_label = str(1 + i)
            G.add_node(node_label, label=node_label) # Add the 'label' attribute

        # Add edges based on the adjacency matrix
        for i in range(len(self.adj_matrix)):
            for j in range(i + 1, len(self.adj_matrix)):

                if self.adj_matrix[i][j] == 1:
                    node_i = str(1 + i)
                    node_j = str(1 + j)
                    G.add_edge(node_i, node_j)

        pos = nx.spring_layout(G, seed=self.s)

        return G, pos

    def transform_positions(self):
        """
        Transforms positions for plotting.

        Returns:
        - transformed_positions: Transformed positions for plotting.
        """
        pos_array = np.array(list(self.pos.values()))
        max_xy = np.max(pos_array, 0)
        min_xy = np.min(pos_array, 0)
        X = (pos_array[:, 0] - min_xy[0]) / (max_xy[0] - min_xy[0])
        Y = (pos_array[:, 1] - min_xy[1]) / (max_xy[1] - min_xy[1])
        pos_ = np.round(np.stack((X, Y), axis=1) * (self.scalar ** (1 / 2)),
                        decimals=3)
        transformed_positions = {v: list(pos) for v,
                                 pos in zip(self.pos.keys(), pos_)}
        return transformed_positions

    def calculate_distances(self, G, pos):
        """
        Calculates distances between nodes in the graph.

        Parameters:
        - G: NetworkX graph.
        - pos: Dictionary mapping nodes to positions in the graph.

        Returns:
        - weights_dict: Dictionary of distances between nodes.
        """
        weights_dict = {}
        memory = []
        for v in G.adj.keys():
            adj_weights = {}
            for v_ in G.adj[v].keys():
                x_1, x_2, y_1, y_2 = \
                    pos[v][0], pos[v_][0], pos[v][1], pos[v_][1]
                distance = \
                    round(((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2) ** (1 / 2), 3)
                adj_weights[v_] = distance
                memory.append((v_, v))
            weights_dict[v] = adj_weights
        return weights_dict

    def add_weights(self, G):
        """
        Adds weights to the graph based on calculated distances.

        Parameters:
        - G: NetworkX graph.

        Returns:
        - G: NetworkX graph with weights added.
        """
        for v in G.adj.keys():
            for v_ in G.adj[v].keys():
                G[v][v_]['weight'] = self.distances[v][v_]
        return G

    def plot_graph(self, node_size=600, node_color='skyblue', edge_color='gray',
                   alpha=0.5, edge_width=1, edges_font_size=6,
                   labels_font_weight='bold', labels_font_color='black',
                   labels_font_size=8):
        """
        Plots the generated graph with specified parameters.

        Parameters:
        - node_size: Size of the nodes.
        - node_color: Color of the nodes.
        - edge_color: Color of the edges.
        - alpha: Transparency of the graph elements.
        - edge_width: Width of the edges.
        - edges_font_size: Font size for edge labels.
        - labels_font_weight: Font weight for node labels.
        - labels_font_color: Font color for node labels.
        - labels_font_size: Font size for node labels.
        """
        plt.figure(figsize=(20, 20))

        # Draw nodes
        nx.draw_networkx_nodes(self.G, self.transformed_positions,
                               node_size=node_size, node_color=node_color,
                               alpha=alpha)

        # Draw edges
        nx.draw_networkx_edges(self.G, self.transformed_positions,
                               edge_color=edge_color, width=edge_width,
                               alpha=alpha)

        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.transformed_positions,
                                     edge_labels=edge_labels,
                                     font_size=edges_font_size,
                                     alpha=alpha)

        # Draw node labels
        labels = nx.get_node_attributes(self.G, 'label')
        nx.draw_networkx_labels(self.G, self.transformed_positions,
                                labels=labels, font_weight=labels_font_weight,
                                font_color=labels_font_color,
                                font_size=labels_font_size)

        plt.show()