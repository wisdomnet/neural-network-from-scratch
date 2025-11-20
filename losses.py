import numpy as np


class Loss:
    """Abstract class for a loss function."""

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError()

    def loss_prime(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def loss_batch(self, y_true_batch: np.ndarray, y_pred_batch: np.ndarray) -> float:
        raise NotImplementedError()

    def loss_prime_batch(self, y_true_batch: np.ndarray, y_pred_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class MSE(Loss):
    """Mean squared error loss."""
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.power(y_true - y_pred, 2))

    def loss_prime(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = np.prod(y_true.shape)
        return 2.0 * (y_pred - y_true) / n

    def loss_batch(self, y_true_batch: np.ndarray, y_pred_batch: np.ndarray) -> float:
        return np.mean(np.power(y_true_batch - y_pred_batch, 2))*y_true_batch.shape[0]

    def loss_prime_batch(self, y_true_batch: np.ndarray, y_pred_batch: np.ndarray) -> np.ndarray:
        n = np.prod(y_true_batch.shape[1:])
        return 2.0 * (y_pred_batch - y_true_batch) / n



class CrossEntropy(Loss):
    """Cross entropy loss."""

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -np.sum(y_true * np.log(y_pred))

    def loss_prime(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -y_true / y_pred

    def loss_batch(self, y_true_batch: np.ndarray, y_pred_batch: np.ndarray) -> float:
        return -np.sum(y_true_batch * np.log(y_pred_batch))
    
    def loss_prime_batch(self, y_true_batch: np.ndarray, y_pred_batch: np.ndarray) -> np.ndarray:
        return -y_true_batch / y_pred_batch 
