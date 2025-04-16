import numpy as np

class DenseLayer:
  def __init__(self, input_size, output_size, activation, activation_deriv, use_adam=False):
    self.weights = np.random.randn(input_size, output_size) * 0.01
    self.biases = np.zeros((1, output_size))
    self.activation = activation
    self.activation_deriv = activation_deriv

    self.use_adam = use_adam
    self.iteration = 0

    if use_adam:
      self.m_w = np.zeros_like(self.weights)
      self.v_w = np.zeros_like(self.weights)
      self.m_b = np.zeros_like(self.biases)
      self.v_b = np.zeros_like(self.biases)

  def forward(self, input_data):
    self.input = input_data
    self.z = np.dot(input_data, self.weights) + self.biases
    self.output = self.activation(self.z)
    return self.output

  def backward(self, output_gradient, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    activation_grad = self.activation_deriv(self.z) * output_gradient
    input_grad = np.dot(activation_grad, self.weights.T)
    weights_grad = np.dot(self.input.T, activation_grad)
    bias_grad = np.sum(activation_grad, axis=0, keepdims=True)

    if self.use_adam:
      self.iteration += 1
      self.m_w = beta1 * self.m_w + (1 - beta1) * weights_grad
      self.v_w = beta2 * self.v_w + (1 - beta2) * (weights_grad ** 2)
      m_w_hat = self.m_w / (1 - beta1 ** self.iteration)
      v_w_hat = self.v_w / (1 - beta2 ** self.iteration)

      self.m_b = beta1 * self.m_b + (1 - beta1) * bias_grad
      self.v_b = beta2 * self.v_b + (1 - beta2) * (bias_grad ** 2)
      m_b_hat = self.m_b / (1 - beta1 ** self.iteration)
      v_b_hat = self.v_b / (1 - beta2 ** self.iteration)

      self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
      self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
    else:
      # Regular gradient descent
      self.weights -= learning_rate * weights_grad
      self.biases -= learning_rate * bias_grad

    return input_grad
