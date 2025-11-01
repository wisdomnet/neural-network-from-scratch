import numpy as np
import modules


class Optimizer:
    """Abstract class for updating the parameters of a module."""

    def __init__(self, module: modules.Module):
        self.module = module

    def step(self):
        raise NotImplementedError()

    def zero_gradients(self):
        for grad in self.module.gradients():
            grad.fill(0)


class SGD(Optimizer):
    def __init__(
        self,
        module: modules.Module,
        learning_rate: float = 0.01,
        momentum: float = 0.9
    ):
        super().__init__(module)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = [np.zeros_like(param) for param in module.parameters()]

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.gradients()
        for i in range(len(parameters)):
            self.v[i] = (
                self.momentum * self.v[i] + self.learning_rate * gradients[i]
            )
            parameters[i] -= self.v[i]


class GradientDescent(Optimizer):
    """
    Batch Gradient Descent optimizer.
    This optimizer accumulates gradients over the entire dataset
    and then performs a single update step using the average gradient.
    """

    def __init__(
        self,
        module: modules.Module,
        learning_rate: float = 0.01
    ):
        super().__init__(module)
        self.learning_rate = learning_rate

    def step(self, num_samples: int = 1):
        """
        Performs a single Batch GD step.
        This method assumes gradients have been accumulated over num_samples.
        It updates parameters using the *average* gradient.
        """
        if num_samples <= 0:
            return  # Avoid division by zero

        parameters = self.module.parameters()
        gradients = self.module.gradients()
        for i in range(len(parameters)):
            # Calculate the average gradient
            avg_grad = gradients[i] / num_samples
            # Apply the update
            parameters[i] -= self.learning_rate * avg_grad
