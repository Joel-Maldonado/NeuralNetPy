import numpy as np

class Activation:
    def __init__(self):
        pass
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return self.__class__.__name__


class ReLU(Activation):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dout[self.x <= 0] = 0
        return dout


class GeLU(Activation):
    def forward(self, x):
        self.x = x
        return x * 0.5 * (1 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
    
    def backward(self, dout):
        return dout * (0.5 * (1 + np.tanh((np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))))) * (1 - 0.5 * np.power(np.tanh((np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))), 2))
    

class Softmax(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
