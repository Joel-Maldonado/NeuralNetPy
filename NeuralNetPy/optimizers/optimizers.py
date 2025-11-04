import numpy as np

class Optimizer:
    def __init__(self):
        pass
    
    def step(self, layers):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.lr * layer.dweights
                layer.biases -= self.lr * layer.dbiases


class Adam:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = [layer for layer in layers if hasattr(layer, 'weights')]
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def step(self):
        if self.m is None:
            self.m = [np.zeros_like(layer.weights) for layer in self.layers]
            self.v = [np.zeros_like(layer.weights) for layer in self.layers]
            self.m_b = [np.zeros_like(layer.biases) for layer in self.layers]
            self.v_b = [np.zeros_like(layer.biases) for layer in self.layers]

        self.t += 1

        for i, layer in enumerate(self.layers):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * layer.dweights
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (layer.dweights ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            weight_update = -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            layer.weights += weight_update

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.dbiases
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.dbiases ** 2)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            bias_update = -self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            layer.biases += bias_update

        # Reset the gradients for the next iteration
        for layer in self.layers:
            layer.dweights = np.zeros_like(layer.weights)
            layer.dbiases = np.zeros_like(layer.biases)
