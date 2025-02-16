import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import metrics

class Perceptron:
    def __init__(self, weights, bias, learning_rate = 0.01, epochs = 10):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epoch_loss = []

    def activate(self, w_sum):
        return 1 / (1 + np.exp(-w_sum))

    def loss(self, target, pred):
        return -(target * np.log10(pred) + (1 - target) * np.log10(1 - pred))

    def predict(self, x):
        return self.activate(np.dot(x, self.weights) + self.bias)

    def classify(self, pred):
        threshold = 0.5
        return 1 if pred >= threshold else 0

    def train(self, data):
        for epoch in range(self.epochs):
            individual_loss = []
            for i in range(len(data)):
                sample = data.iloc[i][:-1]
                target = data.iloc[i][-1]

                pred = self.predict(sample)
                loss = self.loss(target, pred)

                self.weights = self.weights + self.learning_rate * (target - pred) * sample.to_numpy()
                self.bias = self.bias + self.learning_rate * (target - pred)

                individual_loss.append(loss)
            self.epoch_loss.append(sum(individual_loss) / len(data))
        self.plot_loss()

    def test(self, data):
        actual, predicted = [], []
        for i in range(len(data)):
            sample = data.iloc[i][:-1]
            target = data.iloc[i][-1]

            pred = self.predict(sample)
            output = self.classify(pred)

            actual.append(target)
            predicted.append(output)
        self.plot_confusion(np.array(actual), np.array(predicted))

    def plot_loss(self):
        df = pd.DataFrame(self.epoch_loss)
        plot = df.plot(kind='line', grid=True)
        plot.set_xlabel('Epochs')
        plot.set_ylabel('Loss')
        plot.set_title('Training Loss')
        plt.savefig('output/training_loss.pdf')

    def plot_confusion(self, actual, predicted):
        confusion_matrix = metrics.confusion_matrix(actual, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                    display_labels=["Setosa", "Versicolor"])
        cm_display.plot()
        plt.savefig('output/confusion_matrix.pdf')
