import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import functools

import networkx as nx


def get_thresholds_popularity(num_graph, edgedata, graph_ind):
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

        id_location = [index + 1 for index, element in enumerate(graph_ind) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = np.array(edgedata[edgedata['from'].isin(id_location)])

        graph = nx.from_edgelist(graph_edges)

        for v in id_location:
            try:
                I = [n for n in graph.neighbors(v)]
                sum = 0
                for u in I:
                    sum = sum + graph.degree(u)
                popularity = graph.degree(v)+(sum/len(I))
            except:
                continue

            thresh_list.append(popularity)
        thresh_list = list(np.unique(np.array(thresh_list)))

    I1 = np.unique(np.array(thresh_list))

    return I1


def popularity_sub(num_graph, edgedata, graph_ind, threshold_array):
    '''
    This function calculates the vector of number of 0-cells and number of 1-cells for sub-level popularity filtration
    function.The parameters are:

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
        popul = list()

        id_location = [index + 1 for index, element in enumerate(graph_ind) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = np.array(edgedata[edgedata['from'].isin(id_location)])

        graph = nx.from_edgelist(graph_edges)

        for v in id_location:
            n1 = list()
            result = list()
            n1.append(v)

            try:
                I = [n for n in graph.neighbors(v)]
                sum = 0
                for u in I:
                    sum = sum + graph.degree(u)
                popularity = graph.degree(v) + (sum / len(I))
            except:
                continue
            result.append(popularity)

            s = n1 + result
            # list of values for each node
            popul.append(s)

        # Filtration function algorithm
        for degr in threshold_array:
            Rindex = [index for (index, degree) in popul if degree <= degr]

            sub = graph.subgraph(np.array(Rindex))

            c0.append(len(np.unique(Rindex)))
            c1.append(sub.number_of_edges())

        cell0.append(c0)
        cell1.append(c1)

    return cell0, cell1


def popularity_super(num_graph, edgedata, graph_ind, threshold_array):
    '''
    This function calculates the vector of number of 0-cells and number of 1-cells for super-level popularity filtration
    function.The parameters are:

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
        popul = list()

        id_location = [index + 1 for index, element in enumerate(graph_ind) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = np.array(edgedata[edgedata['from'].isin(id_location)])

        graph = nx.from_edgelist(graph_edges)

        for v in id_location:
            n1 = list()
            result = list()
            n1.append(v)

            try:
                I = [n for n in graph.neighbors(v)]
                sum = 0
                for u in I:
                    sum = sum + graph.degree(u)
                popularity = graph.degree(v) + (sum / len(I))
            except:
                continue
            result.append(popularity)

            s = n1 + result
            # list of values for each node
            popul.append(s)

        # Filtration function algorithm
        for degr in threshold_array:
            Rindex = [index for (index, degree) in popul if degree >= degr]

            sub = graph.subgraph(np.array(Rindex))

            c0.append(len(np.unique(Rindex)))
            c1.append(sub.number_of_edges())

        cell0.append(c0)
        cell1.append(c1)

    return cell0, cell1