import numpy as np

class Loss:
    def __init__(self):
        pass
    
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    
    def backward(self, y_pred, y_true):
        raise NotImplementedError


class MSE(Loss):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return np.mean(np.square(diff))

    def backward(self, y_pred, y_true):
        self.dout = 2 * (y_pred - y_true) / y_pred.shape[0]
        return self.dout

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - np.sum(y_true * np.log(y_pred), axis=-1)

    def backward(self, y_pred, y_true):
        self.dout = y_pred - y_true
        return self.dout
