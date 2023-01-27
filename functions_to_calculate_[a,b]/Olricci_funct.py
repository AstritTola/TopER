import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx


def get_thresholds_olricci(num_graph, edgedata, graph_ind):
    '''
    This function calculates the thresholds array. The parameters are:

    input:
    num_graph: int, total number of graphs
    edgedata: nd array, edge list
    graph_ind: array, enumeration from 1 to num_graph

    return:
    array of thresholds

    '''

    thresh_list = list()

    for graph_id in range(1,num_graph + 1):
        ricci=list()

        id_location = [index + 1 for index, element in enumerate(graph_ind) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = np.array(edgedata[edgedata['from'].isin(id_location)])

        graph = nx.from_edgelist(graph_edges)
        orc = OllivierRicci(graph, alpha=0.5, verbose="INFO")
        orc.compute_ricci_curvature()

        for line in np.array(graph_edges):
            ricci.append(round(orc.G[line[0]][line[1]]["ricciCurvature"],6))
        thresh_list = thresh_list + list(ricci)
        thresh_list = list(np.unique(np.array(thresh_list)))

    I1 = np.unique(np.array(thresh_list))

    return I1


def olricci_sub(num_graph, edgedata, graph_ind, threshold_array):
    '''
    This function calculates the vector of number of 0-cells and number of 1-cells for sub-level Ollivier Ricci
    Curvature filtration function.The parameters are:

    input:
    num_graph: int, total number of graphs
    edgedata: nd array, edge list
    graph_ind: array, enumeration from 1 to num_graph
    threshold_array: array of thresholds


    return:
    ndarrays of number of 0-cells and 1-cells vectors

    '''

    threshold_array = sorted(threshold_array)
    cell0 = list()
    cell1 = list()

    for graph_id in range(1, num_graph + 1):
        c0 = list()
        c1 = list()
        ricci = list()

        id_location = [index + 1 for index, element in enumerate(graph_ind) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = np.array(edgedata[edgedata['from'].isin(id_location)])

        graph = nx.from_edgelist(graph_edges)
        orc = OllivierRicci(graph, alpha=0.5, verbose="INFO")
        orc.compute_ricci_curvature()

        for line in np.array(graph_edges):
            n1 = list()
            n2 = list()
            result = list()
            n1.append(line[0])
            n2.append(line[1])
            l = n1 + n2
            result.append(round(orc.G[line[0]][line[1]]["ricciCurvature"],6))
            s = l + result
            # list of values for each edge
            ricci.append(s)

        # Filtration function algorithm
        for degr in threshold_array:
            Rindex = [(n1,n2) for (n1, n2, coeff) in ricci if coeff <= degr]

            sub = graph.edge_subgraph(Rindex)

            c0.append(sub.number_of_nodes())
            c1.append(sub.number_of_edges())

        cell0.append(c0)
        cell1.append(c1)

    return cell0, cell1


def olricci_super(num_graph, edgedata, graph_ind, threshold_array):
    '''
    This function calculates the vector of number of 0-cells and number of 1-cells for super-level Ollivier Ricci
    Curvature filtration function.The parameters are:

    input:
    num_graph: int, total number of graphs
    edgedata: nd array, edge list
    graph_ind: array, enumeration from 1 to num_graph
    threshold_array: array of thresholds


    return:
    ndarrays of number of 0-cells and 1-cells vectors

    '''

    threshold_array = sorted(threshold_array, reverse=True)
    cell0 = list()
    cell1 = list()
    for graph_id in range(1, num_graph + 1):
        c0 = list()
        c1 = list()
        ricci = list()

        id_location = [index + 1 for index, element in enumerate(graph_ind) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = np.array(edgedata[edgedata['from'].isin(id_location)])

        graph = nx.from_edgelist(graph_edges)
        orc = OllivierRicci(graph, alpha=0.5, verbose="INFO")
        orc.compute_ricci_curvature()

        for line in np.array(graph_edges):
            n1 = list()
            n2 = list()
            result = list()
            n1.append(line[0])
            n2.append(line[1])
            l = n1 + n2
            result.append(round(orc.G[line[0]][line[1]]["ricciCurvature"], 6))
            s = l + result
            # list of values for each edge
            ricci.append(s)

        # Filtration function algorithm
        for degr in threshold_array:
            Rindex = [(n1, n2) for (n1, n2, coeff) in ricci if coeff >= degr]

            sub = graph.edge_subgraph(Rindex)

            c0.append(sub.number_of_nodes())
            c1.append(sub.number_of_edges())

        cell0.append(c0)
        cell1.append(c1)

    return cell0, cell1