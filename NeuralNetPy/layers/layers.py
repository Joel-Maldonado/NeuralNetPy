import numpy as np

class Layer:
    def __init__(self):
        self.weights = None
   
    def forward(self, x):
        raise NotImplementedError
   
    def backward(self, grad):
        raise NotImplementedError
   
    def __repr__(self):
        raise NotImplementedError
    
    def __call__(self, x):
        try:
            return self.forward(x)
        except ValueError as e:
            print(f"Expected shape: {self.weights.shape} But received shape: {x.shape}")
            
   

class Dense(Layer):
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
       
        self.weights = np.random.randn(n_outputs, n_inputs) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_outputs))

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)


    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights.T) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
       
        return np.dot(dvalues, self.weights)
       
    def __repr__(self):
        return f"Dense({self.n_inputs}, {self.n_outputs})"


class Flatten(Layer):
    def __init__(self):
        pass
       
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
   
    def backward(self, dvalues):
        self.dout = dvalues.reshape(self.input_shape)
        return self.dout
    
    def __repr__(self):
        return f"Flatten"

class Debug(Layer):
    def __init__(self):
        pass
       
    def forward(self, x):
        print(f"Forward: {x.shape}")
        return x
   
    def backward(self, dvalues):
        print(f"Backward: {dvalues.shape}")
        return dvalues
    


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size) * np.sqrt(1. / (self.kernel_size))
        self.biases = np.random.randn(self.out_channels) * np.sqrt(1. / self.out_channels)

        self.dweights = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.dbiases = np.zeros((self.out_channels))

        self.cache = None

    def forward(self, X):
        in_n, in_c, in_h, in_w = X.shape

        n_C = self.out_channels
        n_H = int((in_h + 2 * self.padding - self.kernel_size)/ self.stride) + 1
        n_W = int((in_w + 2 * self.padding - self.kernel_size)/ self.stride) + 1
        
        X_col = self._im2col(X, self.kernel_size, self.kernel_size, self.stride, self.padding)
        w_col = self.weights.reshape((self.out_channels, -1))
        b_col = self.biases.reshape(-1, 1)
        
        # Perform matrix multiplication.
        out = np.dot(w_col, X_col.T) + b_col
        
        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, in_n)).reshape((in_n, n_C, n_H, n_W))
        
        self.cache = X, X_col, w_col
        return out

    def backward(self, dout):
        X, X_col, w_col = self.cache
        in_n, _, _, _ = X.shape

        self.dbiases = np.sum(dout, axis=(0,2,3))

        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, in_n))
        dout = np.concatenate(dout, axis=-1)

        dX_col = np.dot(w_col.T, dout)
        dw_col = np.dot(dout, X_col)
        dX = self._col2im(dX_col, X.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)

        self.dweights = dw_col.reshape((dw_col.shape[0], self.in_channels, self.kernel_size, self.kernel_size))
                
        return dX
    
    def _im2col(self, input_data, filter_h, filter_w, stride, pad):
        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        N, C, H, W = img.shape
        NN, CC, HH, WW = img.strides
        out_h = (H - filter_h)//stride + 1
        out_w = (W - filter_w)//stride + 1
        col = np.lib.stride_tricks.as_strided(img, (N, out_h, out_w, C, filter_h, filter_w), (NN, stride * HH, stride * WW, CC, HH, WW)).astype(float)
        return col.reshape(np.multiply.reduceat(col.shape, (0, 3)))
    
    def _col2im(self, col, input_shape, filter_h, filter_w, stride, pad):
        N, C, H, W = input_shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        col_reshaped = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
        X_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=col.dtype)
        for i in range(out_h):
            for j in range(out_w):
                h_start, w_start = i * stride, j * stride
                h_end, w_end = h_start + filter_h, w_start + filter_w
                X_padded[:, :, h_start:h_end, w_start:w_end] += col_reshaped[:, i, j]

        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[pad:-pad, pad:-pad, :, :]

    def __repr__(self):
        return f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class MaxPool2D(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.cache = None

    def forward(self, X):
        self.cache = X
        n, c, h, w = X.shape
        ph, pw = self.pool_size
        oh, ow = h//ph, w//pw
        out = np.zeros((n, c, oh, ow))
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = np.amax(X[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw], axis=(2, 3))
        return out

    def backward(self, dout):
        n, c, oh, ow = dout.shape
        ph, pw = self.pool_size
        _, _, h, w = self.cache.shape
        dx = np.zeros_like(self.cache)
        for i in range(oh):
            for j in range(ow):
                window = self.cache[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                m = np.amax(window, axis=(2, 3), keepdims=True)
                mask = (window == m)
                dx[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw] += mask * (dout[:, :, i, j])[:, :, None, None]
        return dx
    
    def __repr__(self):
        return f"MaxPool2D({self.pool_size})"

class AveragePooling2D(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.cache = None
    
    def forward(self, X):
        N, C, self.H, self.W = X.shape
        kh, kw = self.pool_size, self.pool_size
        new_h = int(self.H / kh)
        new_w = int(self.W / kw)
        X_reshaped = X.reshape(N, C, kh, kw, new_h, new_w)
        out = X_reshaped.mean(axis=(2, 3))
        self.cache = (X_reshaped, out)
        return out

    def backward(self, dout):
        X_reshaped, out = self.cache
        N, C, kh, kw, new_h, new_w = X_reshaped.shape
        dX_reshaped = np.zeros_like(X_reshaped)
        dX_reshaped.reshape(N, C, kh*kw, new_h, new_w)[range(N), :, :, :, :] = \
            dout[:, :, np.newaxis, :, :] / (kh*kw)
        dX = dX_reshaped.reshape(N, C, self.H, self.W)
        return dX
    
    def __repr__(self):
        return f"MaxPool2D({self.pool_size})"

