import numpy as np


class Module:
    """Abstract class for a module in a neural network."""

    def __init__(self):
        self.inputs: np.ndarray = None
        self.outputs: np.ndarray = None
        self.inputs_batch: np.ndarray = None
        self.outputs_batch: np.ndarray = None
        self.name: str = self.__class__.__name__

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def parameters(self) -> list[np.ndarray]:
        return []

    def gradients(self) -> list[np.ndarray]:
        return []

    def forward_batch(self, inputs_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward_batch(self, output_grad_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Linear(Module):
    """Linear transformation applied to input column vector."""

    def __init__(self, input_size: int, output_size: int, xavier: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
        if xavier:
            self.weights = self.weights / np.sqrt(input_size)
            self.bias = self.bias / np.sqrt(output_size)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = np.dot(self.weights, inputs) + self.bias
        return self.outputs

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.weights_grad += np.dot(output_grad, self.inputs.T)
        self.bias_grad += output_grad
        return np.dot(self.weights.T, output_grad)

    def parameters(self) -> list[np.ndarray]:
        return [self.weights, self.bias]

    def gradients(self) -> list[np.ndarray]:
        return [self.weights_grad, self.bias_grad]

    def forward_batch(self, inputs_batch: np.ndarray) -> np.ndarray:
        """
        self.inputs_batch's shape: (batch_size, input_size, 1)
        self.weights's shape: (output_size, input_size)
        self.bias's shape: (output_size, 1)
        self.outputs_batch's shape: (batch_size, output_size, 1)
        """
       
        self.inputs_batch = inputs_batch
        self.outputs_batch = (self.weights @ self.inputs_batch) + self.bias
        return self.outputs_batch

    def backward_batch(self, output_grad_batch: np.ndarray) -> np.ndarray:
        inputs_batch_T    = np.transpose(self.inputs_batch, (0, 2, 1))
        self.weights_grad = np.sum(output_grad_batch @ inputs_batch_T, axis=0)
        self.bias_grad    = np.sum(output_grad_batch, axis=0)
        return self.weights.T @ output_grad_batch

       


class Softmax(Module):
    """Softmax activation layer."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = np.exp(inputs)
        self.outputs /= np.sum(self.outputs)
        return self.outputs

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        n = np.size(self.outputs)
        return np.dot((np.identity(n) - self.outputs.T) * self.outputs, output_grad)

    def forward_batch(self, inputs_batch: np.ndarray) -> np.ndarray:
        self.inputs_batch = inputs_batch
        self.outputs_batch = np.exp(inputs_batch)
        self.outputs_batch /= np.sum(self.outputs_batch, axis=1, keepdims=True)
        return self.outputs_batch

    def backward_batch(self, output_grad_batch: np.ndarray) -> np.ndarray:
        n    = np.prod(self.outputs_batch.shape[1:])
        M    = self.outputs_batch
        M_t  = np.transpose(M, (0, 2, 1))
        I_n  = np.identity(n)
        return ((I_n-M_t)*M) @ output_grad_batch


class Activation(Module):
    """Applies an activation function to the input, element wise."""

    def activation(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return self.activation(inputs)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * self.activation_prime(self.inputs)

    def forward_batch(self, inputs_batch: np.ndarray) -> np.ndarray:
        self.inputs_batch = inputs_batch
        return self.activation(inputs_batch)

    def backward_batch(self, output_grad_batch: np.ndarray) -> np.ndarray:
        return output_grad_batch * self.activation_prime(self.inputs_batch)


class Tanh(Activation):
    """Hyperbolic tangent activation layer."""

    def activation(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs)

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        return 1.0 - np.power(np.tanh(inputs), 2)


class Sequential(Module):
    """Sequential neural network."""

    def __init__(self, modules: list[Module]):
        super().__init__()
        self.modules = modules

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs
        for module in self.modules:            
            outputs = module.forward(outputs)
        return outputs

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        input_grad = output_grad
        for module in reversed(self.modules):
            input_grad = module.backward(input_grad)
        return input_grad

    def parameters(self) -> list[np.ndarray]:
        return [param for module in self.modules for param in module.parameters()]

    def gradients(self) -> list[np.ndarray]:
        return [grad for module in self.modules for grad in module.gradients()]

    def forward_batch(self, inputs_batch: np.ndarray) -> np.ndarray:
        outputs_batch = inputs_batch
        for module in self.modules: 
            outputs_batch = module.forward_batch(outputs_batch)
        return outputs_batch

    def backward_batch(self, output_grad_batch: np.ndarray) -> np.ndarray:
        input_grad_batch= output_grad_batch
        for module in reversed(self.modules):
            #print(f'Backwarding through {module.name}')
            #print(f'Input shape: {input_grad_batch.shape}') 
            input_grad_batch = module.backward_batch(input_grad_batch)
            #print(f'Output shape: {input_grad_batch.shape}\n')
        return input_grad_batch