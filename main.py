from src.perceptron import Perceptron
from src.utils import get_data, generate_weights

train_data, test_data = get_data()
weights = generate_weights(train_data.shape[1] - 1)
bias = 0
learning_rate = 0.001
epochs = 100

def main():
    perceptron = Perceptron(weights, bias, learning_rate, epochs)
    perceptron.train(train_data)
    perceptron.test(test_data)

if __name__ == '__main__':
    main()
