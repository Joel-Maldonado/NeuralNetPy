import numpy as np
        
class Dataset:
    def __init__(self, X: np.array, y: np.array, batch_size: int = 32, shuffle: bool = True):
        assert len(X.shape) > 1, "X data must have atleast 2 dims. shape: (n, input_size)"
        assert len(y.shape) > 1, "y data must have atleast 2 dims. shape: (n, input_size)"

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
