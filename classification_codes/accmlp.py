def create_model(units, units2, lr, decay, act):
    model = Sequential()
    model.add(Dense(units, input_dim=4, activation=act, kernel_initializer=GlorotUniform(),
                    kernel_regularizer=regularizers.l2(decay)))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='softmax'))

    if units2 == 'A':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif units2 == 'E':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    return model


def cr_model(units, epoch, units2, bs, decay, act, X_train1, y_train1, X_test1, y_test1):
    param_grid = {
        'units': [units],
        'units2': [units2],
        'epochs': [epoch],
        'batch_size': [bs],
        'lr': [0.01],
        'decay': [decay],
        'act': [act]
    }

    # Create a list to store the models and their accuracies
    models = []

    # Perform parameter grid search
    for params in product(*param_grid.values()):
        hyperparameters = dict(zip(param_grid.keys(), params))

        model = create_model(units=hyperparameters['units'], units2=hyperparameters['units2'],
                             lr=hyperparameters['lr'], decay=hyperparameters['decay'],
                             act=hyperparameters['act'])

        keras_model = KerasClassifier(model=model, epochs=hyperparameters['epochs'],
                                      batch_size=hyperparameters['batch_size'])

        keras_model.fit(X_train1, y_train1)

        y_pred = keras_model.predict(X_test1)
        auc = accuracy_score(y_test1, y_pred)
        models.append((auc, params))

    return models


def param_selection(X_train, y_train, X_val, y_val):
    M1 = len(y_train + len(y_val)) / 19
    M2 = len(y_train + len(y_val)) / 9
    M3 = len(y_train + len(y_val)) / 4
    M4 = len(y_train + len(y_val)) / 20
    M5 = len(y_train + len(y_val)) / 10
    M6 = len(y_train + len(y_val)) / 5
    print(M1, M2)

    units_values = [2, 4, 8, 16]
    units2_value = ['A', 'E']
    epochs_value = [50, 100, 200]
    batch_size_value = [int(M1), int(M2), int(M3), int(M4), int(M5), int(M6)]
    decay_value = [0.001, 1e-5, 1e-6]
    act_fn = ['relu', 'gelu', 'elu']

    results = []

    for units in units_values:
        for units2 in units2_value:
            for epoch in epochs_value:
                for bs in batch_size_value:
                    for decay in decay_value:
                        for act in act_fn:
                            results.append(
                                cr_model(units, epoch, units2, bs, decay, act, X_train, y_train, X_val, y_val))

    best_model1 = max(results, key=lambda x: x[0])

    return best_model1[0]


import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from itertools import product
from tensorflow.keras import regularizers
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from keras.initializers import GlorotUniform
from sklearn.metrics import accuracy_score

with open('DHFR.csv', 'r') as f:
    data = f.readlines()[1:]

d_data = np.array([list(map(float, edge.strip().split('\t')[2:])) for edge in data])

r, c = np.shape(d_data)
print(r, c)

m = int((c - 3) / 2)
print(m)
y = d_data[:, c - 1]
X = d_data[:, :c - 1]

name = 'DHFR'
file = open(name + 'resultsnew3.csv', 'w')
file.write('dataset' + "\t" + 'fun id' + "\t" + 'accuracy' + "\t" + 'std' + "\n")
file.flush()

I = []
for l in range(m):
    for k in range(r):
        if str(X[k, 2 * l]) == "nan":
            I.append(l)
            break

print(I)

for j in range(m):
    if j not in I:
        score = []

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        X1 = X[:, [2 * j, 2 * j + 1, c - 3, c - 2]]

        # Iterate over the folds
        for train_index, test_index in skf.split(X, y):
            # Split the data into train and test sets based on the fold indices
            X_train, X_test = X1[train_index], X1[test_index]
            y_train, y_test = y[train_index], y[test_index]

            accuracy = param_selection(X_train, y_train, X_test, y_test)
            score.append(accuracy)
        print(j)
        print(f"The accuracy with standard deviation: {np.mean(score) * 100:.2f} ± {np.std(score) * 100:.2f} ")

        mean1 = np.mean(score) * 100
        std1 = np.std(score) * 100
        file.write(name + "\t" + str(j) + "\t" + str(mean1) + "\t" + str(std1) + "\n")
        file.flush()
file.close()
print('done')
