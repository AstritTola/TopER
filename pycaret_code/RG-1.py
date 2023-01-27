"""
created on : 23-01-23
description : in this code, I use a dataframe that contains vectors a and b for 7 different functions, # of edges and nodes.
implementation : select best model from pycaret, build a model using default cv(10) and predict on test data
"""
import random
import sys
import csv

import pandas as pd
from pycaret.classification import *
import numpy as np

random.seed(123)


def read_data(dataset):
    df_path = pd.read_csv(datapath + "/" + dataset + ".csv", header=0)
    # drop dataset and graph_id columns
    drop_fn = df_path.drop(df_path.filter(regex='dataset|graph_id').columns, axis=1)

    # drop col if it contains any nan value
    if not drop_fn.isnull().values.any():
        new_df = drop_fn
    else:
        new_df = drop_fn.dropna(axis=1, how="any")

    working_df = new_df.iloc[:, :-3]  # select columns excluding the last three columns
    last_cols = new_df.iloc[:, -3:]  # select last three columns

    return working_df, last_cols


def run_duplication(working_df, last_cols):
    for i in np.arange(0, (working_df.shape[1] // 2)):
        subset_df = working_df.iloc[:,
                    [(2 * i), ((2 * i) + 1)]]  # split df into every two consecutive columns (1, 2), (3, 4) etc
        col_names = ((subset_df.columns[0]).split("_", 1))[1]
        merge_df = pd.concat([subset_df, last_cols], axis=1)  # concatenate function with num. of nodes and edges

        cr_setup = setup(data=merge_df, target='graphlabels', train_size=0.9, normalize=True,
                         normalize_method='minmax')  # setup environment and normalize data
        best_models = compare_models()  # compare baseline models
        best_model = pull().index[0]  # select the index for best model

        # create model using best_model
        # this function trains and evaluates a model using cross validation set by fold
        model = create_model(best_model, round=3)
        pred = predict_model(model)
        pred_acc = (pull().set_index('Model'))['Accuracy'].iloc[0]  # accuracy of prediction using first best model

        lister = [dataset, str(col_names), str(pred_acc), str(best_model)]
        writer.writerow(lister)


if __name__ == "__main__":
    datapath = sys.argv[1]  # data path
    datalist = (
    'BZR', 'COX2', 'MUTAG', 'DHFR', 'PROTEINS', 'IMDB_BINARY', 'IMDB_MULTI', 'REDDIT_BINARY', 'REDDIT_MULTI')
    outputfile = "path to result folder" + "accuracy.csv"
    header = ['dataset', 'col_names', 'pred_acc', 'best_model']
    file = open(outputfile, 'w')
    writer = csv.writer(file)
    writer.writerow(header)
    for dataset in datalist:
        working_df, last_cols = read_data(dataset)
        for dup in np.arange(5):
            run_duplication(working_df, last_cols)
