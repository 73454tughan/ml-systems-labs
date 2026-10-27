import numpy as np
from src.layers import DenseLayer, ReLUActivation, SoftmaxCrossEntropy

class TwoHiddenLayerMLP:
    """
    Two-Hidden-Layer Neural Network Architecture:
    Input (64) -> Dense(64, W) -> ReLU -> Dense(W, W) -> ReLU -> Dense(W, 10) -> Softmax
    Weights ~ N(0, 0.01^2), Biases = 0.
    """
    def __init__(self, input_dim: int = 64, hidden_dim: int = 16, output_dim: int = 10, weight_scale: float = 0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Layer 1
        self.fc1 = DenseLayer(input_dim, hidden_dim, weight_scale)
        self.act1 = ReLUActivation()

        # Layer 2
        self.fc2 = DenseLayer(hidden_dim, hidden_dim, weight_scale)
        self.act2 = ReLUActivation()

        # Layer 3 (Output logits)
        self.fc3 = DenseLayer(hidden_dim, output_dim, weight_scale)
        
        # Loss criterion
        self.criterion = SoftmaxCrossEntropy()

    def forward(self, X: np.ndarray, Y: np.ndarray = None):
        """Executes forward pass through all layers."""
        z1 = self.fc1.forward(X)
        a1 = self.act1.forward(z1)

        z2 = self.fc2.forward(a1)
        a2 = self.act2.forward(z2)

        z3 = self.fc3.forward(a2)

        if Y is not None:
            loss = self.criterion.forward(z3, Y)
            return z3, self.criterion.probs, loss
        else:
            shift_Z = z3 - np.max(z3, axis=1, keepdims=True)
            exp_Z = np.exp(shift_Z)
            probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
            return z3, probs

    def backward(self):
        """Backpropagates gradient from loss through all affine and activation layers."""
        dZ3 = self.criterion.backward()
        dA2 = self.fc3.backward(dZ3)
        dZ2 = self.act2.backward(dA2)
        dA1 = self.fc2.backward(dZ2)
        dZ1 = self.act1.backward(dA1)
        dX = self.fc1.backward(dZ1)
        return dX

    def step(self, lr: float):
        """Applies Gradient Descent step to all trainable parameters."""
        self.fc1.update(lr)
        self.fc2.update(lr)
        self.fc3.update(lr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, Y_onehot: np.ndarray):
        _, probs, loss = self.forward(X, Y_onehot)
        preds = np.argmax(probs, axis=1)
        labels = np.argmax(Y_onehot, axis=1)
        acc = float(np.mean(preds == labels))
        return loss, acc
