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
        

class Dataset:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X))
        self.current_index = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= len(self.X):
            self.current_index = 0
            if self.shuffle:
                np.random.shuffle(self.indexes)
            raise StopIteration
        else:
            start = self.current_index
            end = self.current_index + self.batch_size
            self.current_index += self.batch_size
            return self.X[self.indexes[start:end]], self.y[self.indexes[start:end]]
    
    def __len__(self):
        return len(self.X) // self.batch_size
    
    def __getitem__(self, idx):
        batch_idx = idx * self.batch_size
        return self.X[batch_idx:batch_idx+self.batch_size], self.y[batch_idx:batch_idx+self.batch_size]
        
def one_hot_encode(scalar, num_classes):
    return np.eye(num_classes)[scalar]
    
def acc_fn(y_pred, y_true):
    return (y_pred.argmax(axis=1) == y_true.argmax(axis=1)).mean()
