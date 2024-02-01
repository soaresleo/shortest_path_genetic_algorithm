import numpy as np

def attrib_priorities(chromo_len):
    """
    Generates an array of unique random integers representing priorities.

    Parameters:
    - chromo_len: Length of the chromosome.

    Returns:
    - random_ints: Array of unique random integers.
    """
    ints = np.arange(1, chromo_len + 1, 1)
    random_ints = np.random.choice(ints, chromo_len, replace=False)
    return random_ints

def U_arr_build(n, t_col_index):
    """
    Builds a 2D array with ones, except for a column of zeros.

    Parameters:
    - n: Size of the array.
    - t_col_index: Index of the column with zeros.

    Returns:
    - U_arr: 2D array with ones and a column of zeros.
    """
    U_arr = np.ones((n, n), 'int8')
    U_arr[:, t_col_index] = 0
    return U_arr

def get_eligible_nodes(dyn_adj_arr, t):
    """
    Get eligible nodes at a given time step from a dynamic adjacency array.

    Parameters:
    - dyn_adj_arr: Dynamic adjacency array.
    - t: Time step.

    Returns:
    - eligible_nodes: Array of eligible nodes.
    """
    nonzeros = np.nonzero(np.array(dyn_adj_arr)[t, :])
    return nonzeros[0]

def add_node(path, priorities, eligible_nodes):
    """
    Adds a node to the path based on priorities and eligible nodes.

    Parameters:
    - path: Current path.
    - priorities: Node priorities.
    - eligible_nodes: Eligible nodes for the next step.

    Returns:
    - path: Updated path.
    """
    max_idx = np.argmax(priorities[eligible_nodes])
    path.append(eligible_nodes[max_idx])
    return path

def raise_path(adj_arr, end_node_index, priorities=None):
    """
    Raises a path through the adjacency array until reaching the end node.

    Parameters:
    - adj_arr: Adjacency array.
    - end_node_index: Index of the end node.
    - priorities: Node priorities.

    Returns:
    - priorities: Final node priorities.
    - path: Raised path.
    - has_dead_node: Flag indicating if a dead node is encountered.
    """
    dyn_adj_arr = adj_arr.copy()
    n = len(adj_arr)
    path = [0]
    t = 0
    U_arr = U_arr_build(n, t)
    mesh_arr = np.ones((n, n), 'int8')
    mesh_arr[:, 0] = 0

    if priorities is None:
        priorities = attrib_priorities(n)

    k = 1

    while t != end_node_index:
        eligible_nodes = get_eligible_nodes(dyn_adj_arr, t)

        if eligible_nodes.size == 0:
            # Pendant node treatment required!
            t = end_node_index
            path.append(t)
            return priorities, path, True

        add_node(path, priorities, eligible_nodes)
        t = path[-1]
        U_arr = U_arr_build(n, t)
        mesh_arr = np.multiply(mesh_arr, U_arr)
        dyn_adj_arr = np.multiply(dyn_adj_arr, mesh_arr)
        k += 1

    return priorities, path, False
