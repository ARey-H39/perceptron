import numpy as np
import pandas as pd

def get_data():
    data = pd.read_csv('data/iris_data.csv')
    data['target'] = np.where(data['target'] == "Setosa", 0, 1)

    setosa_train_data = data[data['target'] == 0].head(40)
    versicolor_train_data = data[data['target'] == 1].head(40)

    setosa_test_data = data[data['target'] == 0].tail(10)
    versicolor_test_data = data[data['target'] == 1].tail(10)

    train_data = pd.concat([setosa_train_data, versicolor_train_data])
    test_data = pd.concat([setosa_test_data, versicolor_test_data])

    return train_data, test_data

def generate_weights(n_cols):
    rg = np.random.default_rng()
    weights = rg.random((1, n_cols))[0]
    return weights
