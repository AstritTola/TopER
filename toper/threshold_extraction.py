import numpy as np
import networkx as nx
import joblib
import pandas as pd
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from .utils import *


def get_thresholds_deg_cen(van_graph_list):
    
    thresh = []
    graph_list = []
    for G in van_graph_list:
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(G)
        degree_centrality_values = list(degree_centrality.values())
        thresh = thresh + degree_centrality_values
        
        # Add degree centrality as a node attribute
        nx.set_node_attributes(G, degree_centrality, "degree_centrality")
            
        graph_list.append(G)
    print('done_thresh_degcen')
        
    return graph_list, np.unique(thresh)
        
def get_thresholds_popularity(van_graph_list):
    thresh = []
    graph_list = []

    for G in van_graph_list:

        pop_dict = popularity(G)
        popularity_values = list(pop_dict.values())
        thresh += popularity_values

        nx.set_node_attributes(G, pop_dict, "popularity")
        graph_list.append(G)
    print('done_thresh_popularity')

    return graph_list, np.unique(thresh)

def get_thresholds_closeness(van_graph_list):
    thresh = []
    graph_list = []

    for G in van_graph_list:

        closeness = compute_node_closeness(G)#nx.closeness_centrality(G), list(G.nodes())
        closeness_rounded = {node: val for node, val in closeness.items()}
        closeness_values = list(closeness_rounded.values())
        thresh += closeness_values

        nx.set_node_attributes(G, closeness_rounded, "closeness")
        graph_list.append(G)
    print('done_thresh_closeness')

    return graph_list, np.unique(thresh)


def get_thresholds_forricci(van_graph_list):

    thresh = []
    graph_list = []

    for G in van_graph_list:

        ricci = FormanRicci(G)
        ricci.compute_ricci_curvature()
        ricci_vals = [round(val, 6) for _, _, val in ricci.G.edges.data("formanCurvature")]
        thresh += ricci_vals

        graph_list.append(ricci.G)

    print('done_thresh_fricci')

    return graph_list, np.unique(thresh)

def get_thresholds_olricci(van_graph_list):
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    thresh = []
    graph_list = []

    for G in van_graph_list:

        ricci = OllivierRicci(G, alpha=0.5, verbose="ERROR")
        ricci.compute_ricci_curvature()
        ricci_vals = [round(val, 6) for _, _, val in ricci.G.edges.data("ricciCurvature")]
        thresh += ricci_vals

        graph_list.append(ricci.G)

    print('done_thresh_oricci')

    return graph_list, np.unique(thresh)
    

def get_thresholds_degree(van_graph_list):
    thresh = []
    graph_list = []

    for G in van_graph_list:

        degrees = dict(G.degree())
        degrees_rounded = {node: val for node, val in degrees.items()}
        thresh += list(degrees_rounded.values())

        nx.set_node_attributes(G, degrees_rounded, "degree")
        graph_list.append(G)

    print('done_thresh_degree')

    return graph_list, np.unique(thresh)


def get_thresholds_weight(van_graph_list):
    thresh = []

    for G in van_graph_list:
        # dictionary: {node: weight}
        weight_dict = nx.get_node_attributes(G, "weight")
        thresh.extend(weight_dict.values())

    return van_graph_list, np.unique(thresh)

