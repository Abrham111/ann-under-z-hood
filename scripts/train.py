import numpy as np
from activations import tanh, tanh_derivative, softmax
from layers import DenseLayer
from network import SimpleNeuralNet


def generate_dummy_data(samples=100, features=2, classes=3):
  x = np.random.randn(samples, features)
  y = np.zeros((samples, classes))
  for i in range(samples):
    y[i, np.random.randint(0, classes)] = 1
  return x, y


def main():
  # Set seed for reproducibility
  np.random.seed(1)

  # Generate dummy data
  x_train, y_train = generate_dummy_data()

  # Initialize the network
  net = SimpleNeuralNet()
  net.add(DenseLayer(2, 4, tanh, tanh_derivative, use_adam=True))
  net.add(DenseLayer(4, 3, softmax, lambda x: 1, use_adam=True))  # Output layer with softmax

  # Train the network
  net.train(
    x_train=x_train,
    y_train=y_train,
    epochs=50,
    batch_size=10,
    learning_rate=0.01
  )


if __name__ == "__main__":
  main()
