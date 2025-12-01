from .threshold_extraction import *
from .utils import *

def sub_node_feat(num_graph, G_list, threshold_array, metric_key):
    """
    Computes subgraph feature counts (node and edge count) under varying thresholds
    without constructing subgraphs.

    Parameters:
    - num_graph: int - total number of graphs
    - G_list: list of NetworkX graphs
    - threshold_array: list of threshold values (sorted)
    - metric_key: str - node attribute key (e.g., 'degree_centrality')

    Returns:
    - TopER: list of TopER features
    """
    threshold_array = sorted(threshold_array)

    TopER = []

    for graph_id in range(num_graph):
        graph = G_list[graph_id]
        nodes_metric = nx.get_node_attributes(graph, metric_key)
        edges = list(graph.edges())

        c0 = []
        c1 = []

        for thresh in threshold_array:
            Rindex = {node for node, val in nodes_metric.items() if val <= thresh}
            num_nodes = len(Rindex)
            num_edges = sum(1 for u, v in edges if u in Rindex and v in Rindex)

            c0.append(num_nodes)
            c1.append(num_edges)

        TopER.append(best_fit_u1mean0(c0,c1))

    return TopER

def super_node_feat(num_graph, G_list, threshold_array, metric_key):
    """
    Computes subgraph feature counts (node and edge count) under varying thresholds
    without constructing subgraphs.

    Parameters:
    - num_graph: int - total number of graphs
    - G_list: list of NetworkX graphs
    - threshold_array: list of threshold values (sorted)
    - metric_key: str - node attribute key (e.g., 'degree_centrality')

    Returns:
    - TopER: list of TopER features
    """
    threshold_array = sorted(threshold_array)
 
    TopER = []

    for graph_id in range(num_graph):
        graph = G_list[graph_id]
        nodes_metric = nx.get_node_attributes(graph, metric_key)
        edges = list(graph.edges())

        c0 = []
        c1 = []

        for thresh in threshold_array:
            Rindex = {node for node, val in nodes_metric.items() if val >= thresh}
            num_nodes = len(Rindex)
            num_edges = sum(1 for u, v in edges if u in Rindex and v in Rindex)

            c0.append(num_nodes)
            c1.append(num_edges)

        TopER.append(best_fit_u1mean0(c0,c1))

    return TopER

def sub_edge_feat(num_graph, G_list, threshold_array, metric_key):
    """
    Subgraph PH computation for edge-based metrics (e.g., Ricci curvature).
    Counts nodes and edges where only edges with metric <= threshold are kept.

    Parameters:
    - num_graph: int - total number of graphs
    - G_list: list of NetworkX graphs
    - threshold_array: list of threshold values (sorted)
    - metric_key: str - edge attribute key (e.g., 'formanCurvature', 'ricciCurvature')

    Returns:
    - TopER: list of TopER features
    """
    threshold_array = sorted(threshold_array)

    TopER = []

    for graph_id in range(num_graph):
        graph = G_list[graph_id]
        edges_metric = nx.get_edge_attributes(graph, metric_key)

        c0 = []
        c1 = []

        for thresh in threshold_array:
            # Filter edges based on edge attribute threshold
            filtered_edges = [(u, v) for (u, v), val in edges_metric.items() if val <= thresh]

            # Get all unique nodes that appear in the filtered edges
            involved_nodes = set()
            for u, v in filtered_edges:
                involved_nodes.add(u)
                involved_nodes.add(v)

            c0.append(len(involved_nodes))
            c1.append(len(filtered_edges))

        TopER.append(best_fit_u0mean1(c0,c1))

    return TopER

def super_edge_feat(num_graph, G_list, threshold_array, metric_key):
    """
    Subgraph PH computation for edge-based metrics (e.g., Ricci curvature).
    Counts nodes and edges where only edges with metric <= threshold are kept.

    Parameters:
    - num_graph: int - total number of graphs
    - G_list: list of NetworkX graphs
    - threshold_array: list of threshold values (sorted)
    - metric_key: str - edge attribute key (e.g., 'formanCurvature', 'ricciCurvature')

    Returns:
    - TopER: list of TopER features
    """
    threshold_array = sorted(threshold_array)

    TopER = []

    for graph_id in range(num_graph):
        graph = G_list[graph_id]
        edges_metric = nx.get_edge_attributes(graph, metric_key)

        c0 = []
        c1 = []

        for thresh in threshold_array:
            # Filter edges based on edge attribute threshold
            filtered_edges = [(u, v) for (u, v), val in edges_metric.items() if val >= thresh]

            # Get all unique nodes that appear in the filtered edges
            involved_nodes = set()
            for u, v in filtered_edges:
                involved_nodes.add(u)
                involved_nodes.add(v)

            c0.append(len(involved_nodes))
            c1.append(len(filtered_edges))

        TopER.append(best_fit_u0mean1(c0,c1))

    return TopER

def compute_and_save_features(prefix, num_graph, graph_list, thresholds, func_sub, func_super, mkey):
    F_sub = func_sub(num_graph, graph_list, thresholds, mkey)
    F_super = func_super(num_graph, graph_list, thresholds, mkey)
    
    return F_sub, F_super
