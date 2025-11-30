import networkx as nx
import numpy as np
import pandas as pd
import functools

def popularity(G, as_dict=True):
    """
    Compute a custom 'popularity' metric for each node in the graph.

    popularity(v) = degree(v) + (sum of degrees of neighbors) / (number of neighbors)

    Args:
        G (nx.Graph): A NetworkX graph.
        as_dict (bool): If True, return a dictionary {node: popularity}.
                        If False, return an iterator like G.degree().

    Returns:
        dict or iterator: Popularity values per node.
    """
    pop_values = {}

    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) == 0:
            # Isolated node → only its own degree counts
            pop_values[v] = G.degree(v)
            continue

        neighbor_degrees_sum = sum(G.degree(u) for u in neighbors)
        pop_values[v] = G.degree(v) + neighbor_degrees_sum / len(neighbors)

    if as_dict:
        return pop_values
    else:
        return ((v, val) for v, val in pop_values.items())

def compute_hks(graph, t_values):
    """
    Compute the Heat Kernel Signature (HKS) for each node in the graph.
    :param graph: NetworkX graph (undirected, unweighted)
    :param t_values: List of diffusion time values to compute HKS
    :return: Dictionary with nodes as keys and HKS values as lists
    """
    L = nx.laplacian_matrix(graph).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)  

    hks = {node: [] for node in graph.nodes()}
    
    for t in t_values:
        heat_kernel = np.dot(eigvecs, np.dot(np.diag(np.exp(-t * eigvals)), eigvecs.T))
        for i, node in enumerate(graph.nodes()):
            hks[node].append(heat_kernel[i, i])  
    return hks

def compute_node_closeness(G, u=None, distance=None):
    if G.is_directed():
        G = G.reverse()  # create a reversed graph view

    if distance is not None:
        # use Dijkstra's algorithm with specified attribute as edge weight
        path_length = functools.partial(
            nx.single_source_dijkstra_path_length, weight=distance
        )
    else:
        path_length = nx.single_source_shortest_path_length

    if u is None:
        nodes = G.nodes
    else:
        nodes = [u]
    closeness_dict = {}
    for n in nodes:
        sp = path_length(G, n)
        totsp = sum(sp.values())
        len_G = len(G)
        _closeness_centrality = 0.0
        if totsp > 0.0 and len_G > 1:
            _closeness_centrality = 1 / totsp
        closeness_dict[n] = _closeness_centrality
    if u is not None:
        return closeness_dict[u]
    return closeness_dict

def best_fit_u1mean0(X, Y):
    """
    Compute best-fit line Y ≈ a + bX based on group means of X for each unique Y.
    Returns [a, b].
    """
    # Convert to pandas Series for convenient grouping
    df = pd.DataFrame({'X': X, 'Y': Y})
    
    # Compute mean of X for each unique Y
    grouped = df.groupby('Y', sort=True)['X'].mean().reset_index()
    Yn = grouped['Y'].to_numpy()
    Xn = grouped['X'].to_numpy()

    # Precompute sums (efficient least-squares)
    A = len(Xn)
    B = Xn.sum()
    C = (Xn**2).sum()
    D = (Xn * Yn).sum()
    E = Yn.sum()

    denom = A * C - B**2
    a = (C * E - B * D) / denom
    b = (-B * E + A * D) / denom

    return [a, b]


def best_fit_u0mean1(X, Y):
    """
    Compute best-fit line Y ≈ a + bX based on group means of Y for each unique X.
    Returns [a, b].
    """
    # Build DataFrame for easy grouping
    df = pd.DataFrame({'X': X, 'Y': Y})

    # Compute mean of Y for each unique X
    grouped = df.groupby('X', sort=True)['Y'].mean().reset_index()
    Xn = grouped['X'].to_numpy()
    Yn = grouped['Y'].to_numpy()

    # Precompute sums (least-squares)
    A = len(Xn)
    B = Xn.sum()
    C = (Xn**2).sum()
    D = (Xn * Yn).sum()
    E = Yn.sum()

    denom = A * C - B**2
    a = (C * E - B * D) / denom
    b = (-B * E + A * D) / denom

    return [a, b]


    
