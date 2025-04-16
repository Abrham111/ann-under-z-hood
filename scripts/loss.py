import numpy as np

# Loss function and derivative (cross-entropy)
def cross_entropy_loss(predictions, targets):
  m = targets.shape[0]
  clipped_preds = np.clip(predictions, 1e-12, 1 - 1e-12)
  return -np.sum(targets * np.log(clipped_preds)) / m

def cross_entropy_derivative(predictions, targets):
  return predictions - targets  # derivative of softmax + cross-entropy