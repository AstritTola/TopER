import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import functools

import networkx as nx


def get_thresholds_weight(num_graph, edgedata, graph_ind, nodelabels):
    '''
    This function calculates the thresholds array. The parameters are:

    input:
    num_graph: int, total number of graphs
    edgedata: nd array, edge list
    graph_ind: array, enumeration from 1 to num_graph
    nodelabels: array retrieved from *_node_labels or *_node_attributes

    return:
    array of thresholds

    '''

    I1 = np.unique(np.array(np.unique(np.array(nodelabels))))


    return I1


def weight_sub(num_graph, edgedata, graph_ind, threshold_array, nodelabels):
    '''
    This function calculates the vector of number of 0-cells and number of 1-cells for sub-level weight filtration
    function.The parameters are:

    input:
    num_graph: int, total number of graphs
    edgedata: nd array, edge list
    graph_ind: array, enumeration from 1 to num_graph
    threshold_array: array of thresholds
    nodelabels: array retrieved from *_node_labels or *_node_attributes


    return:
    ndarrays of number of 0-cells and 1-cells vectors

    '''

    threshold_array = sorted(threshold_array)
    cell0 = list()
    cell1 = list()
    nodelabels = np.array(nodelabels)

    for graph_id in range(1, num_graph + 1):
        c0 = list()
        c1 = list()
        wgt = list()

        id_location = [index + 1 for index, element in enumerate(graph_ind) if
                           element == graph_id]  # list the index of the graph_id locations
        graph_edges = np.array(edgedata[edgedata['from'].isin(id_location)])

        graph = nx.from_edgelist(graph_edges)
        if len(threshold_array) != 0:
            for v in id_location:
                n1 = list()
                result = list()
                n1.append(v)

                result.append(nodelabels[v-1, 0])

                s = n1 + result
                # list of values for each node
                wgt.append(s)

            # Filtration function algorithm
            for degr in threshold_array:
                Rindex = [index for (index, degree) in wgt if degree <= degr]

                sub = graph.subgraph(np.array(Rindex))

                c0.append(len(np.unique(Rindex)))
                c1.append(sub.number_of_edges())
        # Record 0 if the thresholds list is empty
        else:
            c0.append(0)
            c1.append(0)

        cell0.append(c0)
        cell1.append(c1)

    return cell0, cell1


def weight_super(num_graph, edgedata, graph_ind, threshold_array, nodelabels):
    '''
    This function calculates the vector of number of 0-cells and number of 1-cells for super-level weight filtration
    function.The parameters are:

    input:
    num_graph: int, total number of graphs
    edgedata: nd array, edge list
    graph_ind: array, enumeration from 1 to num_graph
    threshold_array: array of thresholds
    nodelabels: array retrieved from *_node_labels or *_node_attributes


    return:
    ndarrays of number of 0-cells and 1-cells vectors

    '''

    threshold_array = sorted(threshold_array, reverse=True)
    cell0 = list()
    cell1 = list()
    nodelabels = np.array(nodelabels)

    for graph_id in range(1, num_graph + 1):
        c0 = list()
        c1 = list()
        wgt = list()

        id_location = [index + 1 for index, element in enumerate(graph_ind) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = np.array(edgedata[edgedata['from'].isin(id_location)])

        graph = nx.from_edgelist(graph_edges)
        if len(threshold_array) != 0:
            for v in id_location:
                n1 = list()
                result = list()
                n1.append(v)

                result.append(nodelabels[v-1, 0])

                s = n1 + result
                # list of values for each node
                wgt.append(s)

            # Filtration function algorithm
            for degr in threshold_array:
                Rindex = [index for (index, degree) in wgt if degree >= degr]

                sub = graph.subgraph(np.array(Rindex))

                c0.append(len(np.unique(Rindex)))
                c1.append(sub.number_of_edges())
        # Record 0 if the thresholds list is empty
        else:
            c0.append(0)
            c1.append(0)

        cell0.append(c0)
        cell1.append(c1)

    return cell0, cell1