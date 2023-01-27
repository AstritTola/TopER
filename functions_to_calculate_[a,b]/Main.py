import pandas as pd
import numpy as np
import scipy
import networkx as nx
from closens import *
from deg_funct import *
from Olricci_funct import *
from popularity import *
from Formricci import *
from deg_cent import *
from weight_funct import *
from reduction import *
import time

'''This code is the main part to calculate the vectors X (nr. of 0-cells) and Y (nr. of 1-cells), denoted as cell0 
and cell1 throughout the code, for functions: 

1. degree centrality 
2. popularity 
3. closeness 
4. Forman Ricci Curvature 
5. Ollivier Ricci Curvature 
6. degree 
7. filtration according node_attributes for Proteins or node_labels 
for the other datasets (which we called wight function for simplicity)

The code firstly evaluates the thresholds array for each filtration function. Then if length of thresholds array is 
less or equal than 500, we keep the original array, otherwise we reduce the number of its terms to 500 by using the 
reduction code. For last function, if the file *_node_labels does not exists, the threshold array is set to be empty.

As a last step, for each function, for both sub-level and super-level filtration, the code calculates and saves in 
respective .csv files the vectors X and Y.



'''



start = time.time()

name = 'REDDIT-BINARY'
edgedata = pd.read_csv(r"......" + name + "/" + name + "_A.txt", header=None)
edgedata.columns = ['from', 'to']
graph_indicators = pd.read_csv(r"......."+name+"/"+name+"_graph_indicator.txt", header=None)
graph_indicators.columns = ["ID"]
graph_ind = (graph_indicators["ID"].values.astype(int))

num_graph = len(np.unique(np.array(graph_indicators))) # total number of graphs

# degree centrality
# Calculation of thresholds
threshold_array_deg_cen = get_thresholds_deg_cen(num_graph, edgedata, graph_ind)

# Reduction of number of thresholds
if len(threshold_array_deg_cen)<=500:
    threshold_array_deg_cen=threshold_array_deg_cen
else:
    threshold_array_deg_cen=reduction(threshold_array_deg_cen)

# Sub-level filtration
features_deg_cen_sub = np.array(deg_cen_sub(num_graph, edgedata, graph_ind, threshold_array_deg_cen))
cell0_deg_cen_sub = pd.DataFrame(features_deg_cen_sub[0, :, :])
cell1_deg_cen_sub = pd.DataFrame(features_deg_cen_sub[1, :, :])
cell0_deg_cen_sub.to_csv(name + "cell0_deg_cen_sub.csv")
cell1_deg_cen_sub.to_csv(name + "cell1_deg_cen_sub.csv")

# delete the vectors to free up the memory
cell0_deg_cen_sub = 0
cell1_deg_cen_sub = 0

# Super-level filtration
features_deg_cen_super = np.array(deg_cen_super(num_graph, edgedata, graph_ind, threshold_array_deg_cen))
cell0_deg_cen_super = pd.DataFrame(features_deg_cen_super[0, :, :])
cell1_deg_cen_super = pd.DataFrame(features_deg_cen_super[1, :, :])
cell0_deg_cen_super.to_csv(name + "cell0_deg_cen_super.csv")
cell1_deg_cen_super.to_csv(name + "cell1_deg_cen_super.csv")

# delete the vectors to free up the memory
cell0_deg_cen_super = 0
cell1_deg_cen_super = 0


# popularity
# Calculation of thresholds
threshold_array_popularity = get_thresholds_popularity(num_graph, edgedata, graph_ind)

# Reduction of number of thresholds
if len(threshold_array_popularity)<=500:
    threshold_array_popularity=threshold_array_popularity
else:
    threshold_array_popularity=reduction(threshold_array_popularity)

# Sub-level filtration
features_popularity_sub = np.array(popularity_sub(num_graph, edgedata, graph_ind, threshold_array_popularity))
cell0_popularity_sub = pd.DataFrame(features_popularity_sub[0, :, :])
cell1_popularity_sub = pd.DataFrame(features_popularity_sub[1, :, :])
cell0_popularity_sub.to_csv(name + "cell0_popularity_sub.csv")
cell1_popularity_sub.to_csv(name + "cell1_popularity_sub.csv")

# delete the vectors to free up the memory
cell0_popularity_sub = 0
cell1_popularity_sub = 0

# Super-level filtration
features_popularity_super = np.array(popularity_super(num_graph, edgedata, graph_ind, threshold_array_popularity))
cell0_popularity_super = pd.DataFrame(features_popularity_super[0, :, :])
cell1_popularity_super = pd.DataFrame(features_popularity_super[1, :, :])
cell0_popularity_super.to_csv(name + "cell0_popularity_super.csv")
cell1_popularity_super.to_csv(name + "cell1_popularity_super.csv")

# delete the vectors to free up the memory
cell0_popularity_super = 0
cell1_popularity_super = 0


# closeness
# Calculation of thresholds
threshold_array_closeness = get_thresholds_closeness(num_graph, edgedata, graph_ind)

# Reduction of number of thresholds
if len(threshold_array_closeness)<=500:
    threshold_array_closeness=threshold_array_closeness
else:
    threshold_array_closeness=reduction(threshold_array_closeness)

# Sub-level filtration
features_closeness_sub = np.array(closeness_sub(num_graph, edgedata, graph_ind, threshold_array_closeness))
cell0_closeness_sub = pd.DataFrame(features_closeness_sub[0, :, :])
cell1_closeness_sub = pd.DataFrame(features_closeness_sub[1, :, :])
cell0_closeness_sub.to_csv(name + "cell0_closeness_sub.csv")
cell1_closeness_sub.to_csv(name + "cell1_closeness_sub.csv")

# delete the vectors to free up the memory
cell0_closeness_sub = 0
cell1_closeness_sub = 0

# Super-level filtration
features_closeness_super = np.array(closeness_super(num_graph, edgedata, graph_ind, threshold_array_closeness))
cell0_closeness_super = pd.DataFrame(features_closeness_super[0, :, :])
cell1_closeness_super = pd.DataFrame(features_closeness_super[1, :, :])
cell0_closeness_super.to_csv(name + "cell0_closeness_super.csv")
cell1_closeness_super.to_csv(name + "cell1_closeness_super.csv")

# delete the vectors to free up the memory
cell0_closeness_super = 0
cell1_closeness_super = 0

# Forman Ricci
# Calculation of thresholds
threshold_array_forricci= get_thresholds_forricci(num_graph, edgedata, graph_ind)

# Reduction of number of thresholds
if len(threshold_array_forricci)<=500:
    threshold_array_forricci=threshold_array_forricci
else:
    threshold_array_forricci=reduction(threshold_array_forricci)

# Sub-level filtration
features_forricci_sub = np.array(forricci_sub(num_graph, edgedata, graph_ind, threshold_array_forricci))
cell0_forricci_sub = pd.DataFrame(features_forricci_sub[0, :, :])
cell1_forricci_sub = pd.DataFrame(features_forricci_sub[1, :, :])
cell0_forricci_sub.to_csv(name + "cell0_forricci_sub.csv")
cell1_forricci_sub.to_csv(name + "cell1_forricci_sub.csv")

# delete the vectors to free up the memory
cell0_forricci_sub = 0
cell1_forricci_sub = 0

# Super-level filtration
features_forricci_super = np.array(forricci_super(num_graph, edgedata, graph_ind, threshold_array_forricci))
cell0_forricci_super = pd.DataFrame(features_forricci_super[0, :, :])
cell1_forricci_super = pd.DataFrame(features_forricci_super[1, :, :])
cell0_forricci_super.to_csv(name + "cell0_forricci_super.csv")
cell1_forricci_super.to_csv(name + "cell1_forricci_super.csv")

# delete the vectors to free up the memory
cell0_forricci_super = 0
cell1_forricci_super = 0

# Ollivier ricci
# Calculation of thresholds
threshold_array_olricci= get_thresholds_olricci(num_graph, edgedata, graph_ind)

# Reduction of number of thresholds
if len(threshold_array_olricci)<=500:
    threshold_array_olricci=threshold_array_olricci
else:
    threshold_array_olricci=reduction(threshold_array_olricci)

# Sub-level filtration
features_olricci_sub = np.array(olricci_sub(num_graph, edgedata, graph_ind, threshold_array_olricci))
cell0_olricci_sub = pd.DataFrame(features_olricci_sub[0, :, :])
cell1_olricci_sub = pd.DataFrame(features_olricci_sub[1, :, :])
cell0_olricci_sub.to_csv(name + "cell0_olricci_sub.csv")
cell1_olricci_sub.to_csv(name + "cell1_olricci_sub.csv")

# delete the vectors to free up the memory
cell0_olricci_sub = 0
cell1_olricci_sub = 0

# Super-level filtration
features_olricci_super = np.array(olricci_super(num_graph, edgedata, graph_ind, threshold_array_olricci))
cell0_olricci_super = pd.DataFrame(features_olricci_super[0, :, :])
cell1_olricci_super = pd.DataFrame(features_olricci_super[1, :, :])
cell0_olricci_super.to_csv(name + "cell0_olricci_super.csv")
cell1_olricci_super.to_csv(name + "cell1_olricci_super.csv")

# delete the vectors to free up the memory
cell0_olricci_super = 0
cell1_olricci_super = 0


# degree
# Calculation of thresholds
threshold_array_degree = get_thresholds_degree(num_graph, edgedata, graph_ind) # thresholds for degree function

# Reduction of number of thresholds
if len(threshold_array_degree)<=500:
    threshold_array_degree=threshold_array_degree
else:
    threshold_array_degree=reduction(threshold_array_degree)

# Sub-level filtration
features_degree_sub = np.array(degree_sub(num_graph, edgedata, graph_ind, threshold_array_degree))
cell0_degree_sub = pd.DataFrame(features_degree_sub[0, :, :])
cell1_degree_sub = pd.DataFrame(features_degree_sub[1, :, :])
cell0_degree_sub.to_csv(name + "cell0_degree_sub.csv")
cell1_degree_sub.to_csv(name + "cell1_degree_sub.csv")

# delete the vectors to free up the memory
cell0_degree_sub = 0
cell1_degree_sub = 0

# Superlevel filtration
features_degree_super = np.array(degree_super(num_graph, edgedata, graph_ind, threshold_array_degree))
cell0_degree_super = pd.DataFrame(features_degree_super[0, :, :])
cell1_degree_super = pd.DataFrame(features_degree_super[1, :, :])
cell0_degree_super.to_csv(name + "cell0_degree_super.csv")
cell1_degree_super.to_csv(name + "cell1_degree_super.csv")

# delete the vectors to free up the memory
cell0_degree_super = 0
cell1_degree_super = 0

# node_labels or node_labels
if name == 'PROTEINS':
    nodelabels=np.array(pd.read_csv(r"....." + name + "/" + name + "_node_attributes.txt", header=None))
    # Calculation of thresholds
    threshold_array_weight = get_thresholds_weight(num_graph, edgedata, graph_ind, nodelabels)

    # Sub-level filtration
    features_weight_sub = np.array(weight_sub(num_graph, edgedata, graph_ind, threshold_array_weight, nodelabels))
    cell0_weight_sub = pd.DataFrame(features_weight_sub[0, :, :])
    cell1_weight_sub = pd.DataFrame(features_weight_sub[1, :, :])
    cell0_weight_sub.to_csv(name + "cell0_weight_sub.csv")
    cell1_weight_sub.to_csv(name + "cell1_weight_sub.csv")

    # Super-level filtration
    features_weight_super = np.array(weight_super(num_graph, edgedata, graph_ind, threshold_array_weight, nodelabels))
    cell0_weight_super = pd.DataFrame(features_weight_super[0, :, :])
    cell1_weight_super = pd.DataFrame(features_weight_super[1, :, :])
    cell0_weight_super.to_csv(name + "cell0_weight_super.csv")
    cell1_weight_super.to_csv(name + "cell1_weight_super.csv")

else:
    try:
        nodelabels = np.array(
            pd.read_csv(r"......" + name + "/" + name + "_node_labels.txt", header=None))
        # Calculation of thresholds
        threshold_array_weight = get_thresholds_weight(num_graph, edgedata, graph_ind,
                                                       nodelabels)
    except:
        nodelabels = []
        threshold_array_weight = []

    # Sub-level filtration
    features_weight_sub = np.array(weight_sub(num_graph, edgedata, graph_ind, threshold_array_weight, nodelabels))
    cell0_weight_sub = pd.DataFrame(features_weight_sub[0, :, :])
    cell1_weight_sub = pd.DataFrame(features_weight_sub[1, :, :])
    cell0_weight_sub.to_csv(name + "cell0_weight_sub.csv")
    cell1_weight_sub.to_csv(name + "cell1_weight_sub.csv")

    # Super-level filtration
    features_weight_super = np.array(weight_super(num_graph, edgedata, graph_ind, threshold_array_weight, nodelabels))
    cell0_weight_super = pd.DataFrame(features_weight_super[0, :, :])
    cell1_weight_super = pd.DataFrame(features_weight_super[1, :, :])
    cell0_weight_super.to_csv(name + "cell0_weight_super.csv")
    cell1_weight_super.to_csv(name + "cell1_weight_super.csv")








print('done')

end = time.time()
print(str(end - start)+' sec')
print(str((end - start) / 60)+' min')
print(str((end - start) / 3600)+' h')
