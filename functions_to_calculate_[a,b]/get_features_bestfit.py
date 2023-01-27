import numpy as np
import pandas as pd
import networkx as nx


# General function
def best_fit_rep(X, Y):
    n = len(X)
    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    for i in range(n):
        A = A + 1
        B = B + X[i]
        C = C + X[i] * X[i]
        D = D + X[i] * Y[i]
        E = E + Y[i]

    a = (C * E - B * D) / (A * C - B * B)
    b = (-B * E + A * D) / (A * C - B * B)
    M = [a, b]

    return M


# These functions must be used in cases when node classification filtration functions are employed to retrieve cell0/cell1 functions
def best_fit_u0(X, Y):
    Xn = np.unique(X)
    n = len(X)

    I = list()
    for u in Xn:
        I1 = list()
        for k in range(n):
            if X[k] == u:
                I1.append(Y[k])
        I.append(np.unique(I1))
    I = np.array(I)

    n1 = len(Xn)
    Yn = np.zeros(n1)

    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    for i in range(n1):
        Yn[i] = I[i, 0]
        A = A + 1
        B = B + Xn[i]
        C = C + Xn[i] * Xn[i]
        D = D + Xn[i] * Yn[i]
        E = E + Yn[i]
    a = (C * E - B * D) / (A * C - B * B)
    b = (-B * E + A * D) / (A * C - B * B)
    M = [a, b]

    return M


def best_fit_u1max0(X, Y):
    Yn = np.unique(Y)
    n = len(X)

    I = list()
    for u in Yn:
        I1 = list()
        for k in range(n):
            if Y[k] == u:
                I1.append(X[k])
            I3 = np.array(I1)
            I2 = max(I3, default=None)
        I.append(np.unique(I2))
    I = np.array(I)

    n1 = len(Yn)
    Xn = np.zeros(n1)

    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    for i in range(n1):
        Xn[i] = I[i, 0]
        A = A + 1
        B = B + Xn[i]
        C = C + Xn[i] * Xn[i]
        D = D + Xn[i] * Yn[i]
        E = E + Yn[i]
    a = (C * E - B * D) / (A * C - B * B)
    b = (-B * E + A * D) / (A * C - B * B)
    M = [a, b]

    return M


def best_fit_u1min0(X, Y):
    Yn = np.unique(Y)
    n = len(X)

    I = list()
    for u in Yn:
        I1 = list()
        for k in range(n):
            if Y[k] == u:
                I1.append(X[k])
            I3 = np.array(I1)
            I2 = min(I3, default=None)
        I.append(np.unique(I2))
    I = np.array(I)

    n1 = len(Yn)
    Xn = np.zeros(n1)

    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    for i in range(n1):
        Xn[i] = I[i, 0]
        Euln = Xn[i] - Yn[i]
        A = A + 1
        B = B + Xn[i]
        C = C + Xn[i] * Xn[i]
        D = D + Xn[i] * Yn[i]
        E = E + Yn[i]
    a = (C * E - B * D) / (A * C - B * B)
    b = (-B * E + A * D) / (A * C - B * B)
    M = [a, b]

    return M


def best_fit_u1mean0(X, Y):
    Yn = np.unique(Y)
    n = len(X)

    I = list()
    for u in Yn:
        I1 = list()
        for k in range(n):
            if Y[k] == u:
                I1.append(X[k])
            I3 = np.array(I1)
            I2 = np.mean(I3)
        I.append(np.unique(I2))
    I = np.array(I)

    n1 = len(Yn)
    Xn = np.zeros(n1)

    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    for i in range(n1):
        Xn[i] = I[i, 0]
        A = A + 1
        B = B + Xn[i]
        C = C + Xn[i] * Xn[i]
        D = D + Xn[i] * Yn[i]
        E = E + Yn[i]
    a = (C * E - B * D) / (A * C - B * B)
    b = (-B * E + A * D) / (A * C - B * B)
    M = [a, b]

    return M


# These functions must be used in cases when edge classification filtration functions are employed to retrieve cell0/cell1 functions
def best_fit_u1(X, Y):
    Yn = np.unique(Y)
    n = len(Y)

    I = list()
    for u in Yn:
        I1 = list()
        for k in range(n):
            if Y[k] == u:
                I1.append(X[k])
        I.append(np.unique(I1))
    I = np.array(I)

    n1 = len(Yn)
    Xn = np.zeros(n1)

    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    for i in range(n1):
        Xn[i] = I[i, 0]
        A = A + 1
        B = B + Xn[i]
        C = C + Xn[i] * Xn[i]
        D = D + Xn[i] * Yn[i]
        E = E + Yn[i]
    a = (C * E - B * D) / (A * C - B * B)
    b = (-B * E + A * D) / (A * C - B * B)
    M = [a, b]

    return M


def best_fit_u0max1(X, Y):
    Xn = np.unique(X)
    n = len(X)

    I = list()
    for u in Xn:
        I1 = list()
        for k in range(n):
            if X[k] == u:
                I1.append(Y[k])
            I3 = np.array(I1)
            I2 = max(I3, default=None)
        I.append(np.unique(I2))
    I = np.array(I)

    n1 = len(Xn)
    Yn = np.zeros(n1)

    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    for i in range(n1):
        Yn[i] = I[i, 0]
        A = A + 1
        B = B + Xn[i]
        C = C + Xn[i] * Xn[i]
        D = D + Xn[i] * Yn[i]
        E = E + Yn[i]
    a = (C * E - B * D) / (A * C - B * B)
    b = (-B * E + A * D) / (A * C - B * B)
    M = [a, b]

    return M


def best_fit_u0min1(X, Y):
    Xn = np.unique(X)
    n = len(X)

    I = list()
    for u in Xn:
        I1 = list()
        for k in range(n):
            if X[k] == u:
                I1.append(Y[k])
            I3 = np.array(I1)
            I2 = min(I3, default=None)
        I.append(np.unique(I2))
    I = np.array(I)

    n1 = len(Xn)
    Yn = np.zeros(n1)

    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    for i in range(n1):
        Yn[i] = I[i, 0]
        A = A + 1
        B = B + Xn[i]
        C = C + Xn[i] * Xn[i]
        D = D + Xn[i] * Yn[i]
        E = E + Yn[i]
    a = (C * E - B * D) / (A * C - B * B)
    b = (-B * E + A * D) / (A * C - B * B)
    M = [a, b]

    return M


def best_fit_u0mean1(X, Y):
    Xn = np.unique(X)
    n = len(X)

    I = list()
    for u in Xn:
        I1 = list()
        for k in range(n):
            if X[k] == u:
                I1.append(Y[k])
            I3 = np.array(I1)
            I2 = np.mean(I3)
        I.append(np.unique(I2))
    I = np.array(I)

    n1 = len(Xn)
    Yn = np.zeros(n1)

    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    for i in range(n1):
        Yn[i] = I[i, 0]
        A = A + 1
        B = B + Xn[i]
        C = C + Xn[i] * Xn[i]
        D = D + Xn[i] * Yn[i]
        E = E + Yn[i]
    a = (C * E - B * D) / (A * C - B * B)
    b = (-B * E + A * D) / (A * C - B * B)
    M = [a, b]

    return M


def counting(graph_ind, graph_id, edgedata):
    id_location = [index + 1 for index, element in enumerate(graph_ind) if
                   element == graph_id]  # list the index of the graph_id locations
    graph_edges = np.array(edgedata[edgedata['from'].isin(id_location)])

    graph = nx.from_edgelist(graph_edges)
    A = graph.number_of_nodes()
    B = graph.number_of_edges()

    return A, B



