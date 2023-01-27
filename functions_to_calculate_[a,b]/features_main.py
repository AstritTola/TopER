import pandas as pd
import numpy as np
import time
from get_features_bestfit import *

'''
In this algorithm the vectors [a, b] are calculated and saved in .cvs file for all functions.
Consider the python file get_features_bestfit to learn about functions.
'''

start = time.time()
name = 'REDDIT-BINARY'

edgedata = pd.read_csv(r"......" + name + "/" + name + "_A.txt", header=None)
edgedata.columns = ['from', 'to']
graph_indicators = pd.read_csv(r"........"+name+"/"+name+"_graph_indicator.txt", header=None)
graph_indicators.columns = ["ID"]
graph_ind = (graph_indicators["ID"].values.astype(int))

file = open(name + '.csv', 'w')
y = pd.read_csv(r".........." + name + "/" + name + "_graph_labels.txt", header=None)
y = np.array(y)
num_graph = len(np.array(y))

file.write('dataset' + "\t" + 'graph_id' +
               "\t" + 'a_degcentsub_all' + "\t" + 'b_degcentsub_all' + "\t" + 'a_degcentsub_u0'  + "\t" + 'b_degcentsub_u0' + "\t" + 'a_degcentsub_u1max0' + "\t" + 'b_degcentsub_u1max0' + "\t" + 'a_degcentsub_u1mean0' + "\t" + 'b_degcentsub_u1mean0' + "\t" + 'a_degcentsub_u1min0' + "\t" + 'b_degcentsub_u1min0' +
               "\t" + 'a_degcentsuper_all' + "\t" + 'b_degcentsuper_all' + "\t" + 'a_degcentsuper_u0'  + "\t" + 'b_degcentsuper_u0' + "\t" + 'a_degcentsuper_u1max0' + "\t" + 'b_degcentsuper_u1max0' + "\t" + 'a_degcentsuper_u1mean0' + "\t" + 'b_degcentsuper_u1mean0' + "\t" + 'a_degcentsuper_u1min0' + "\t" + 'b_degcentsuper_u1min0' +
               "\t" + 'a_popularitysub_all' + "\t" + 'b_popularitysub_all' + "\t" + 'a_popularitysub_u0'  + "\t" + 'b_popularitysub_u0' + "\t" + 'a_popularitysub_u1max0' + "\t" + 'b_popularitysub_u1max0' + "\t" + 'a_popularitysub_u1mean0' + "\t" + 'b_popularitysub_u1mean0' + "\t" + 'a_popularitysub_u1min0' + "\t" + 'b_popularitysub_u1min0' +
               "\t" + 'a_popularitysuper_all' + "\t" + 'b_popularitysuper_all' + "\t" + 'a_popularitysuper_u0'  + "\t" + 'b_popularitysuper_u0' + "\t" + 'a_popularitysuper_u1max0' + "\t" + 'b_popularitysuper_u1max0' + "\t" + 'a_popularitysuper_u1mean0' + "\t" + 'b_popularitysuper_u1mean0' + "\t" + 'a_popularitysuper_u1min0' + "\t" + 'b_popularitysuper_u1min0' +
               "\t" + 'a_closnsub_all' + "\t" + 'b_closnsub_all' + "\t" + 'a_closnsub_u0'  + "\t" + 'b_closnsub_u0' + "\t" + 'a_closnsub_u1max0' + "\t" + 'b_closnsub_u1max0' + "\t" + 'a_closnsub_u1mean0' + "\t" + 'b_closnsub_u1mean0' + "\t" + 'a_closnsub_u1min0' + "\t" + 'b_closnsub_u1min0' +
               "\t" + 'a_closnsuper_all' + "\t" + 'b_closnsuper_all' + "\t" + 'a_closnsuper_u0'  + "\t" + 'b_closnsuper_u0' + "\t" + 'a_closnsuper_u1max0' + "\t" + 'b_closnsuper_u1max0' + "\t" + 'a_closnsuper_u1mean0' + "\t" + 'b_closnsuper_u1mean0' + "\t" + 'a_closnsuper_u1min0' + "\t" + 'b_closnsuper_u1min0' +
               "\t" + 'a_Friccisub_all' + "\t" + 'b_Friccisub_all' + "\t" + 'a_Friccisub_u0'  + "\t" + 'b_Friccisub_u0' + "\t" + 'a_Friccisub_u1max0' + "\t" + 'b_Friccisub_u1max0' + "\t" + 'a_Friccisub_u1mean0' + "\t" + 'b_Friccisub_u1mean0' + "\t" + 'a_Friccisub_u1min0' + "\t" + 'b_Friccisub_u1min0' +
               "\t" + 'a_Friccisuper_all' + "\t" + 'b_Friccisuper_all' + "\t" + 'a_Friccisuper_u0'  + "\t" + 'b_Friccisuper_u0' + "\t" + 'a_Friccisuper_u1max0' + "\t" + 'b_Friccisuper_u1max0' + "\t" + 'a_Friccisuper_u1mean0' + "\t" + 'b_Friccisuper_u1mean0' + "\t" + 'a_Friccisuper_u1min0' + "\t" + 'b_Friccisuper_u1min0' +
               "\t" + 'a_Oriccisub_all' + "\t" + 'b_Oriccisub_all' + "\t" + 'a_Oriccisub_u0'  + "\t" + 'b_Oriccisub_u0' + "\t" + 'a_Oriccisub_u1max0' + "\t" + 'b_Oriccisub_u1max0' + "\t" + 'a_Oriccisub_u1mean0' + "\t" + 'b_Oriccisub_u1mean0' + "\t" + 'a_Oriccisub_u1min0' + "\t" + 'b_Oriccisub_u1min0' +
               "\t" + 'a_Oriccisuper_all' + "\t" + 'b_Oriccisuper_all' + "\t" + 'a_Oriccisuper_u0'  + "\t" + 'b_Oriccisuper_u0' + "\t" + 'a_Oriccisuper_u1max0' + "\t" + 'b_Oriccisuper_u1max0' + "\t" + 'a_Oriccisuper_u1mean0' + "\t" + 'b_Oriccisuper_u1mean0' + "\t" + 'a_Oriccisuper_u1min0' + "\t" + 'b_Oriccisuper_u1min0' +
               "\t" + 'a_degreesub_all' + "\t" + 'b_degreesub_all' + "\t" + 'a_degreesub_u0'  + "\t" + 'b_degreesub_u0' + "\t" + 'a_degreesub_u1max0' + "\t" + 'b_degreesub_u1max0' + "\t" + 'a_degreesub_u1mean0' + "\t" + 'b_degreesub_u1mean0' + "\t" + 'a_degreesub_u1min0' + "\t" + 'b_degreesub_u1min0' +
               "\t" + 'a_degreesuper_all' + "\t" + 'b_degreesuper_all' + "\t" + 'a_degreesuper_u0'  + "\t" + 'b_degreesuper_u0' + "\t" + 'a_degreesuper_u1max0' + "\t" + 'b_degreesuper_u1max0' + "\t" + 'a_degreesuper_u1mean0' + "\t" + 'b_degreesuper_u1mean0' + "\t" + 'a_degreesuper_u1min0' + "\t" + 'b_degreesuper_u1min0' +
               "\t" + 'a_weightsub_all' + "\t" + 'b_weightsub_all' + "\t" + 'a_weightsub_u0'  + "\t" + 'b_weightsub_u0' + "\t" + 'a_weightsub_u1max0' + "\t" + 'b_weightsub_u1max0' + "\t" + 'a_weightsub_u1mean0' + "\t" + 'b_weightsub_u1mean0' + "\t" + 'a_weightsub_u1min0' + "\t" + 'b_weightsub_u1min0' +
               "\t" + 'a_weightsuper_all' + "\t" + 'b_weightsuper_all' + "\t" + 'a_weightsuper_u0'  + "\t" + 'b_weightsuper_u0' + "\t" + 'a_weightsuper_u1max0' + "\t" + 'b_weightsuper_u1max0' + "\t" + 'a_weightsuper_u1mean0' + "\t" + 'b_weightsuper_u1mean0' + "\t" + 'a_weightsuper_u1min0' + "\t" + 'b_weightsuper_u1min0' +
               "\t" + 'nr.nodes' + "\t" + 'nr.edges' + "\t" + 'graphlabels' + "\n")




# degree centrality
cell0_deg_cen_sub = np.array(pd.read_csv(name + "cell0_deg_cen_sub.csv"))[:,1:]
cell1_deg_cen_sub = np.array(pd.read_csv(name + "cell1_deg_cen_sub.csv"))[:,1:]

cell0_deg_cen_super = np.array(pd.read_csv(name + "cell0_deg_cen_super.csv"))[:,1:]
cell1_deg_cen_super = np.array(pd.read_csv(name + "cell1_deg_cen_super.csv"))[:,1:]


# popularity
cell0_popularity_sub = np.array(pd.read_csv(name + "cell0_popularity_sub.csv"))[:,1:]
cell1_popularity_sub = np.array(pd.read_csv(name + "cell1_popularity_sub.csv"))[:,1:]

cell0_popularity_super = np.array(pd.read_csv(name + "cell0_popularity_super.csv"))[:,1:]
cell1_popularity_super = np.array(pd.read_csv(name + "cell1_popularity_super.csv"))[:,1:]


# closeness
cell0_closeness_sub = np.array(pd.read_csv(name + "cell0_closeness_sub.csv"))[:,1:]
cell1_closeness_sub = np.array(pd.read_csv(name + "cell1_closeness_sub.csv"))[:,1:]

cell0_closeness_super = np.array(pd.read_csv(name + "cell0_closeness_super.csv"))[:,1:]
cell1_closeness_super = np.array(pd.read_csv(name + "cell1_closeness_super.csv"))[:,1:]


# Forman Ricci
cell0_forricci_sub = np.array(pd.read_csv(name + "cell0_forricci_sub.csv"))[:,1:]
cell1_forricci_sub = np.array(pd.read_csv(name + "cell1_forricci_sub.csv"))[:,1:]

cell0_forricci_super = np.array(pd.read_csv(name + "cell0_forricci_super.csv"))[:,1:]
cell1_forricci_super = np.array(pd.read_csv(name + "cell1_forricci_super.csv"))[:,1:]


# Ollivier ricci
cell0_olricci_sub = np.array(pd.read_csv(name + "cell0_olricci_sub.csv"))[:,1:]
cell1_olricci_sub = np.array(pd.read_csv(name + "cell1_olricci_sub.csv"))[:,1:]

cell0_olricci_super = np.array(pd.read_csv(name + "cell0_olricci_super.csv"))[:,1:]
cell1_olricci_super = np.array(pd.read_csv(name + "cell1_olricci_super.csv"))[:,1:]


# degree
cell0_degree_sub = np.array(pd.read_csv(name + "cell0_degree_sub.csv"))[:,1:]
cell1_degree_sub = np.array(pd.read_csv(name + "cell1_degree_sub.csv"))[:,1:]

cell0_degree_super = np.array(pd.read_csv(name + "cell0_degree_super.csv"))[:,1:]
cell1_degree_super = np.array(pd.read_csv(name + "cell1_degree_super.csv"))[:,1:]

# weight
cell0_weight_sub = np.array(pd.read_csv(name + "cell0_weight_sub.csv"))[:,1:]
cell1_weight_sub = np.array(pd.read_csv(name + "cell1_weight_sub.csv"))[:,1:]

cell0_weight_super = np.array(pd.read_csv(name + "cell0_weight_super.csv"))[:,1:]
cell1_weight_super = np.array(pd.read_csv(name + "cell1_weight_super.csv"))[:,1:]


for graph_id in range(num_graph):
    features = list()

    # degree centrality sub-level
    features.append(best_fit_rep(cell0_deg_cen_sub[graph_id], cell1_deg_cen_sub[graph_id]))
    features.append(best_fit_u0(cell0_deg_cen_sub[graph_id], cell1_deg_cen_sub[graph_id]))
    features.append(best_fit_u1max0(cell0_deg_cen_sub[graph_id], cell1_deg_cen_sub[graph_id]))
    features.append(best_fit_u1min0(cell0_deg_cen_sub[graph_id], cell1_deg_cen_sub[graph_id]))
    features.append(best_fit_u1mean0(cell0_deg_cen_sub[graph_id], cell1_deg_cen_sub[graph_id]))

    # degree centrality super-level
    features.append(best_fit_rep(cell0_deg_cen_super[graph_id], cell1_deg_cen_super[graph_id]))
    features.append(best_fit_u0(cell0_deg_cen_super[graph_id], cell1_deg_cen_super[graph_id]))
    features.append(best_fit_u1max0(cell0_deg_cen_super[graph_id], cell1_deg_cen_super[graph_id]))
    features.append(best_fit_u1min0(cell0_deg_cen_super[graph_id], cell1_deg_cen_super[graph_id]))
    features.append(best_fit_u1mean0(cell0_deg_cen_super[graph_id], cell1_deg_cen_super[graph_id]))

    # popularity super-level
    features.append(best_fit_rep(cell0_popularity_sub[graph_id], cell1_popularity_sub[graph_id]))
    features.append(best_fit_u0(cell0_popularity_sub[graph_id], cell1_popularity_sub[graph_id]))
    features.append(best_fit_u1max0(cell0_popularity_sub[graph_id], cell1_popularity_sub[graph_id]))
    features.append(best_fit_u1min0(cell0_popularity_sub[graph_id], cell1_popularity_sub[graph_id]))
    features.append(best_fit_u1mean0(cell0_popularity_sub[graph_id], cell1_popularity_sub[graph_id]))

    # popularity super-level
    features.append(best_fit_rep(cell0_popularity_super[graph_id], cell1_popularity_super[graph_id]))
    features.append(best_fit_u0(cell0_popularity_super[graph_id], cell1_popularity_super[graph_id]))
    features.append(best_fit_u1max0(cell0_popularity_super[graph_id], cell1_popularity_super[graph_id]))
    features.append(best_fit_u1min0(cell0_popularity_super[graph_id], cell1_popularity_super[graph_id]))
    features.append(best_fit_u1mean0(cell0_popularity_super[graph_id], cell1_popularity_super[graph_id]))

    # closeness sub-level
    features.append(best_fit_rep(cell0_closeness_sub[graph_id], cell1_closeness_sub[graph_id]))
    features.append(best_fit_u0(cell0_closeness_sub[graph_id], cell1_closeness_sub[graph_id]))
    features.append(best_fit_u1max0(cell0_closeness_sub[graph_id], cell1_closeness_sub[graph_id]))
    features.append(best_fit_u1min0(cell0_closeness_sub[graph_id], cell1_closeness_sub[graph_id]))
    features.append(best_fit_u1mean0(cell0_closeness_sub[graph_id], cell1_closeness_sub[graph_id]))

    # closeness super-level
    features.append(best_fit_rep(cell0_closeness_super[graph_id], cell1_closeness_super[graph_id]))
    features.append(best_fit_u0(cell0_closeness_super[graph_id], cell1_closeness_super[graph_id]))
    features.append(best_fit_u1max0(cell0_closeness_super[graph_id], cell1_closeness_super[graph_id]))
    features.append(best_fit_u1min0(cell0_closeness_super[graph_id], cell1_closeness_super[graph_id]))
    features.append(best_fit_u1mean0(cell0_closeness_super[graph_id], cell1_closeness_super[graph_id]))

    # degree sub-level
    features.append(best_fit_rep(cell0_degree_sub[graph_id], cell1_degree_sub[graph_id]))
    features.append(best_fit_u0(cell0_degree_sub[graph_id], cell1_degree_sub[graph_id]))
    features.append(best_fit_u1max0(cell0_degree_sub[graph_id], cell1_degree_sub[graph_id]))
    features.append(best_fit_u1min0(cell0_degree_sub[graph_id], cell1_degree_sub[graph_id]))
    features.append(best_fit_u1mean0(cell0_degree_sub[graph_id], cell1_degree_sub[graph_id]))


    # degree super-level
    features.append(best_fit_rep(cell0_degree_super[graph_id], cell1_degree_super[graph_id]))
    features.append(best_fit_u0(cell0_degree_super[graph_id], cell1_degree_super[graph_id]))
    features.append(best_fit_u1max0(cell0_degree_super[graph_id], cell1_degree_super[graph_id]))
    features.append(best_fit_u1min0(cell0_degree_super[graph_id], cell1_degree_super[graph_id]))
    features.append(best_fit_u1mean0(cell0_degree_super[graph_id], cell1_degree_super[graph_id]))

    # Forman Ricci sub-level
    features.append(best_fit_rep(cell0_forricci_sub[graph_id], cell1_forricci_sub[graph_id]))
    features.append(best_fit_u1(cell0_forricci_sub[graph_id], cell1_forricci_sub[graph_id]))
    features.append(best_fit_u0max1(cell0_forricci_sub[graph_id], cell1_forricci_sub[graph_id]))
    features.append(best_fit_u0min1(cell0_forricci_sub[graph_id], cell1_forricci_sub[graph_id]))
    features.append(best_fit_u0mean1(cell0_forricci_sub[graph_id], cell1_forricci_sub[graph_id]))

    # Forman Ricci super-level
    features.append(best_fit_rep(cell0_forricci_super[graph_id], cell1_forricci_super[graph_id]))
    features.append(best_fit_u1(cell0_forricci_super[graph_id], cell1_forricci_super[graph_id]))
    features.append(best_fit_u0max1(cell0_forricci_super[graph_id], cell1_forricci_super[graph_id]))
    features.append(best_fit_u0min1(cell0_forricci_super[graph_id], cell1_forricci_super[graph_id]))
    features.append(best_fit_u0mean1(cell0_forricci_super[graph_id], cell1_forricci_super[graph_id]))

    # Ollivier ricci
    features.append(best_fit_rep(cell0_olricci_sub[graph_id], cell1_olricci_sub[graph_id]))
    features.append(best_fit_u1(cell0_olricci_sub[graph_id], cell1_olricci_sub[graph_id]))
    features.append(best_fit_u0max1(cell0_olricci_sub[graph_id], cell1_olricci_sub[graph_id]))
    features.append(best_fit_u0min1(cell0_olricci_sub[graph_id], cell1_olricci_sub[graph_id]))
    features.append(best_fit_u0mean1(cell0_olricci_sub[graph_id], cell1_olricci_sub[graph_id]))

    # Ollivier ricci
    features.append(best_fit_rep(cell0_olricci_super[graph_id], cell1_olricci_super[graph_id]))
    features.append(best_fit_u1(cell0_olricci_super[graph_id], cell1_olricci_super[graph_id]))
    features.append(best_fit_u0max1(cell0_olricci_super[graph_id], cell1_olricci_super[graph_id]))
    features.append(best_fit_u0min1(cell0_olricci_super[graph_id], cell1_olricci_super[graph_id]))
    features.append(best_fit_u0mean1(cell0_olricci_super[graph_id], cell1_olricci_super[graph_id]))

    # weight sub
    features.append(best_fit_rep(cell0_weight_sub[graph_id], cell1_weight_sub[graph_id]))
    features.append(best_fit_u0(cell0_weight_sub[graph_id], cell1_weight_sub[graph_id]))
    features.append(best_fit_u1max0(cell0_weight_sub[graph_id], cell1_weight_sub[graph_id]))
    features.append(best_fit_u1min0(cell0_weight_sub[graph_id], cell1_weight_sub[graph_id]))
    features.append(best_fit_u1mean0(cell0_weight_sub[graph_id], cell1_weight_sub[graph_id]))

    # weight super
    features.append(best_fit_rep(cell0_weight_super[graph_id], cell1_weight_super[graph_id]))
    features.append(best_fit_u0(cell0_weight_super[graph_id], cell1_weight_super[graph_id]))
    features.append(best_fit_u1max0(cell0_weight_super[graph_id], cell1_weight_super[graph_id]))
    features.append(best_fit_u1min0(cell0_weight_super[graph_id], cell1_weight_super[graph_id]))
    features.append(best_fit_u1mean0(cell0_weight_super[graph_id], cell1_weight_super[graph_id]))

    file.write(name + "\t" + str(graph_id + 1) + "\t")
    features = np.array(features)
    num = len(features)

    for k in range(num):
        file.write(str(features[k,0]) + "\t" + str(features[k,1]) + "\t")

    file.write(str(np.array(counting(graph_ind,graph_id+1,edgedata))[0]) + "\t" + str(np.array(counting(graph_ind,graph_id+1,edgedata))[1]) + "\t" + str(y[graph_id,0]) + "\n")

file.close()



print('done')

end = time.time()
print(str(end - start)+' sec')
print(str((end - start) / 60)+' min')
print(str((end - start) / 3600)+' h')