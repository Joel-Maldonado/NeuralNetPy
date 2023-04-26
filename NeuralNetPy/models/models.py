import numpy as np
import os

class BaseModel:
    def __init__(self):
        self.layers = []
   
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
   
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
   
    def add(self, layer):
        self.layers.append(layer)

    def save(self, filename):
        data = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                data[f"layer_{i}_weights"] = layer.weights
                data[f"layer_{i}_biases"] = layer.biases
               
        data['summary'] = self.summary()
        np.savez(filename, **data)
       
    def load(self, filename):
        if os.path.exists(filename):
            data = np.load(filename)
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'weights'):
                    layer.weights = data[f"layer_{i}_weights"]
                    layer.biases = data[f"layer_{i}_biases"]
            return data['summary']
        else:
            raise FileNotFoundError(f"No file found at {filename}")
   
    def state_dict(self):
        data = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                data[f"layer_{i}_weights"] = layer.weights
                data[f"layer_{i}_biases"] = layer.biases
        return data
   
    def load_state_dict(self, data):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                layer.weights = data[f"layer_{i}_weights"]
                layer.biases = data[f"layer_{i}_biases"]
        return data['summary']
       
    def summary(self):
        return "\n".join([str(layer) for layer in self.layers])
    
    @staticmethod
    def load_summary(file):
        return np.load(file)['summary']