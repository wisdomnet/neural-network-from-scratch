import numpy as np
import losses
import modules
import optimizers


def train(
    module: modules.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    loss: losses.Loss,
    optimizer: optimizers.Optimizer,
    epochs: int,
) -> list[float]:
    """Performs training on the given data, loss, and optimizer."""
    errors = []
    for i in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            output = module.forward(x)
            error += loss.loss(y, output)
            module.backward(loss.loss_prime(y, output))
            optimizer.step()
            optimizer.zero_gradients()

        error /= len(x_train)
        errors.append(error)
        print(f"{i+1}/{epochs} error={error:.5f}")
    return errors

def train_batch_gd(
    module: modules.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    loss: losses.Loss,
    optimizer: optimizers.Optimizer,
    epochs: int,
) -> list[float]:
    """
    Performs training using Batch Gradient Descent.
    Updates parameters *per epoch* using the average gradient.
    """
    errors = []
    num_samples = len(x_train)
    for i in range(epochs):
        error = 0
        optimizer.zero_gradients()  # Zero gradients ONCE per epoch

        # 1. Accumulate gradients over the entire dataset
        for x, y in zip(x_train, y_train):
            output = module.forward(x)
            error += loss.loss(y, output)
            # Gradients are accumulated in the module's .backward()
            module.backward(loss.loss_prime(y, output))

        # 2. Update parameters ONCE using the averaged gradients
        optimizer.step(num_samples)  # Pass num_samples for averaging

        error /= num_samples
        errors.append(error)
        print(f"{i+1}/{epochs} error={error:.5f}")
    return errors


def evaluate(
    module: modules.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    loss: losses.Loss,
) -> float:
    """Evaluates the network on the given data and loss."""
    error = sum(loss.loss(y, module.forward(x))
                for x, y in zip(x_test, y_test))
    return error / len(x_test)
