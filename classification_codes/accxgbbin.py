def train_and_evaluate(X_train, y_train, X_test, y_test, max_depth, learning_rate, subsample, colsample_bytree,
                       n_estimators, lamb, obj):
    # initialize the XGBoost classifier
    xgb_clf = xgb.XGBClassifier()

    # set the hyperparameters for this iteration
    xgb_clf.set_params(
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_estimators=n_estimators,
        reg_lambda=lamb,
        objective=obj['objective'],
        eval_metric=obj['eval_metric']
    )

    # train the classifier on the training split
    xgb_clf.fit(X_train, y_train)

    # compute the accuracy on the validation split
    yp = xgb_clf.predict(X_test)
    score = accuracy_score(y_test, yp)

    print(score, max_depth, learning_rate, subsample, colsample_bytree, n_estimators, lamb, obj)
    return score, {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'n_estimators': n_estimators,
        'lambda': lamb,
        'objective': obj['objective'],
        'eval_metric': obj['eval_metric']
    }


def prediction_hits(X_train, y_train, X_test, y_test):
    num_classes = len(np.unique(y_train))

    params_binary = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }

    params_regression = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    param_grid = {
        'max_depth': [1, 3, 5],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.1, 0.5, 0.8, 1],
        'colsample_bytree': [0.1, 0.5, 0.8, 1],
        'n_estimators': [100, 200, 300, 500],
        'lambda': [0.1, 1.0, 10.0],
        'objective': [params_binary, params_regression]
    }

    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            for subsample in param_grid['subsample']:
                for colsample_bytree in param_grid['colsample_bytree']:
                    for n_estimators in param_grid['n_estimators']:
                        for lamb in param_grid['lambda']:
                            for obj in param_grid['objective']:
                                results = train_and_evaluate(X_train, y_train, X_test, y_test, max_depth, learning_rate,
                                                             subsample, colsample_bytree,
                                                             n_estimators, lamb, obj)

    # find the best score and best hyperparameters
    best_score, best_params = max(results, key=lambda x: x[0])

    # train the final model on the full training data with the best hyperparameters
    print(best_score)
    print(best_params)

    return best_score


import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

with open('BZR.csv', 'r') as f:
    data = f.readlines()[1:]

d_data = np.array([list(map(float, edge.strip().split('\t')[2:])) for edge in data])

r, c = np.shape(d_data)
print(r, c)

m = int((c - 3) / 2)
print(m)
y = d_data[:, c - 1]

for W in range(r):
    y[W] = y[W] - 1

X = d_data[:, :c - 1]

name = 'BZR'
file = open(name + 'resultsxgbext.csv', 'w')
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

            accuracy = prediction_hits(X_train, y_train, X_test, y_test)
            score.append(accuracy)

        print(j)
        print(f"The accuracy with standard deviation: {np.mean(score) * 100:.2f} ± {np.std(score) * 100:.2f} ")

        mean1 = np.mean(score) * 100
        std1 = np.std(score) * 100
        file.write(name + "\t" + str(j) + "\t" + str(mean1) + "\t" + str(std1) + "\n")
        file.flush()
file.close()
print('done')

