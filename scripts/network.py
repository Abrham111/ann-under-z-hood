import numpy as np
from loss import cross_entropy_loss, cross_entropy_derivative

class SimpleNeuralNet:
  def __init__(self):
    self.layers = []

  def add(self, layer):
    self.layers.append(layer)

  def forward(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def backward(self, loss_grad, learning_rate):
    for layer in reversed(self.layers):
      loss_grad = layer.backward(loss_grad, learning_rate)

  def train(self, x_train, y_train, epochs, batch_size, learning_rate):
    for epoch in range(epochs):
      permutation = np.random.permutation(len(x_train))
      x_shuffled = x_train[permutation]
      y_shuffled = y_train[permutation]

      for i in range(0, len(x_train), batch_size):
        x_batch = x_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        predictions = self.forward(x_batch)
        loss = cross_entropy_loss(predictions, y_batch)
        grad = cross_entropy_derivative(predictions, y_batch)
        self.backward(grad, learning_rate)

      print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
