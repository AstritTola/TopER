'''
This code uses Pycaret library to perform the machine learning experiment documented in the relative growth rate paper.
The dataframe used are the vectors a and b associated with the superlevel_all function in each dataset folder.
'''


import random
import sys
import csv
import pandas as pd
from pycaret.classification import *
import numpy as np

random.seed(123)


def read_data(dataset):
    df_path = pd.read_csv(datapath + "/" + dataset + ".csv", header=0)  # read data from folder
    # filter dataframe using the substrings in regex
    drop_fn = df_path.filter(regex='super_all|nr.nodes|nr.edges|graphlabels')

    # drop column if it contains any nan value
    if not drop_fn.isnull().values.any():
        new_df = drop_fn
    else:
        new_df = drop_fn.dropna(axis=1, how="any")

    working_df = new_df.iloc[:, :-3]  # select features excluding the last three columns
    last_cols = new_df.iloc[:, -3:]   # select last three columns (no of nodes, edges and graph labels)

    return working_df, last_cols


def run_duplication(working_df, last_cols):
    col_list = ["degcent", "popularity", "closn", "Fricci", "Oricci", "degree", "weight"]
    for z in col_list:
        for j in col_list:
            if z != j:
                fn_1 = working_df.filter(like=z)  # returns vectors a and b attached to function z
                fn_2 = working_df.filter(like=j)  # returns vectors a and b attached to function j
                if len(fn_1.columns) != 0 and len(fn_2.columns) != 0:
                    merge_df = pd.concat([fn_1, fn_2, last_cols], axis=1)

                    #  set up pycaret environment
                    cr_setup = setup(data=merge_df, target='graphlabels', train_size=0.9, normalize=True, normalize_method='minmax')# setup environment and normalize data
                    best_models = compare_models()  # compare baseline models and return mean accuracies
                    best_model = pull().index[0]  # select the index for best model from the df output

                    # create model using first best_model
                    model = create_model(best_model)  # create model using the best model returned
                    pred = predict_model(model) #  predict on test data obtained from the train-test-split in setup function
                    pred_acc = (pull().set_index('Model'))['Accuracy'].iloc[0]  # select accuracy

                    lister = [dataset, str(z), str(j), str(pred_acc), str(best_model)]
                    writer.writerow(lister)


if __name__ == "__main__":
    datapath = sys.argv[1]  # data path
    datalist = ('BZR', 'COX2', 'MUTAG', 'DHFR', 'PROTEINS', 'IMDB_BINARY', 'IMDB_MULTI', 'REDDIT_BINARY', 'REDDIT_MULTI')
    outputfile = "path to result folder" + "accuracy.csv"
    header = ['dataset', 'function_1', 'function_2', 'pred_acc', 'best_model']
    file = open(outputfile, 'w')
    writer = csv.writer(file)
    writer.writerow(header)
    for dataset in datalist:
        working_df, last_cols = read_data(dataset)
        for dup in np.arange(5):
            run_duplication(working_df, last_cols)
