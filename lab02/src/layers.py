import numpy as np

class DenseLayer:
    """
    Fully Connected (Affine) Layer:
    Computes Z = X @ W + b in forward pass.
    Initializes W ~ N(0, 0.01^2), b = 0.
    """
    def __init__(self, in_features: int, out_features: int, weight_scale: float = 0.01):
        self.W = np.random.normal(0.0, weight_scale, (in_features, out_features))
        self.b = np.zeros(out_features)
        
        # Gradients cache
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.X_cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X_cache = X
        return X @ self.W + self.b

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        # dW = X^T @ dZ
        self.dW = self.X_cache.T @ dZ
        # db = sum along batch axis
        self.db = np.sum(dZ, axis=0)
        # dX = dZ @ W^T
        return dZ @ self.W.T

    def update(self, lr: float):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLUActivation:
    """
    Rectified Linear Unit Activation:
    f(z) = max(0, z)
    """
    def __init__(self):
        self.Z_cache = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.Z_cache = Z
        return np.maximum(0.0, Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA * (self.Z_cache > 0.0)


class SoftmaxCrossEntropy:
    """
    Numerically Stable Softmax & Cross-Entropy Loss:
    Softmax: p_k = exp(z_k - max(Z)) / sum(exp(z_j - max(Z)))
    Loss: L = - (1/N) * sum_i sum_k y_{ik} * log(p_{ik} + eps)
    Gradient: dZ = (1/N) * (P - Y)
    """
    def __init__(self, eps: float = 1e-15):
        self.eps = eps
        self.probs = None
        self.targets = None

    def forward(self, Z: np.ndarray, Y: np.ndarray) -> float:
        # Stable softmax using max subtraction
        shift_Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(shift_Z)
        self.probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        self.targets = Y

        # Ensure probabilities sum to 1 within 1e-6
        row_sums = np.sum(self.probs, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Softmax probabilities do not sum to 1 within 1e-6!"

        # Cross-entropy loss
        N = Z.shape[0]
        loss = -np.sum(Y * np.log(self.probs + self.eps)) / N
        return float(loss)

    def backward(self) -> np.ndarray:
        N = self.targets.shape[0]
        # Combined analytical gradient: dL/dZ = (P - Y) / N
        return (self.probs - self.targets) / N
