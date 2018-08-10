import numpy as np

def cnn_forward_naive(X, K, stride=1, pad_width=0):
    """
    Function attributes:
    x: A mini-batch of dimensions (N, C_in, H_in, W_in) where:
        * N is batch size
        * C_in is number of incoming channels
        * H_in is height of each channel
        * W_in is width of each channel
    K: Kernel weights of dimensions (C_out, C_in, KH, KW) where:
        * C_out is the number of outgoing channels
        * H_k is height of the Kernel
        * W_k is width of the Kernel
    """
    N, C_in, H_in, W_in = X.shape
    C_out, _, H_k, W_k = K.shape
    
    H_out = (H_in + 2*pad_width - H_k)//stride + 1
    W_out = (W_in + 2*pad_width - W_k)//stride + 1

    X_padded = zero_pad(X, pad_width=pad_width)
    out = np.zeros((N, C_out, H_out, W_out))
    K_flat = K.reshape(C_out, -1)
    for i in range(H_out):
        for j in range(W_out):
            out[:,:,i,j] = np.matmul(X_padded[:,:,i*stride:(i*stride+H_k), j*stride:(j*stride+W_k)].reshape(N, C_in*H_k*W_k), K_flat.T)
    return out

def zero_pad(X, pad_width=2):
    """
    Input: X, a matrix of shape (N, C, H, W)
    Output: X_padded, a matrix of shape (N, C, H+2*pad_width, W+2*pad_width)

    """
    X_padded = np.pad(X, pad_width=((0,0), (0,0), 2*(pad_width,), 2*(pad_width,)), \
            mode='constant')
    return X_padded

def im2col_slowest(X, K, stride):
    """
    Input: 
        * X, image matrix of shape (N, C_in, H_in, W_in)
        * K, kernel matrix of shape (C_out, C_in, H_k, W_k)
    Output: 
        * im2col matrix of shape (N, C_in,  H_k*W_k, H_out*W_out) where H_out, W_out
          are dimensions of convolution of X with K.

    """
    N, C_in, H_in, W_in = X.shape
    C_out, _, H_k, W_k = K.shape
    H_out = int((H_in - H_k)/stride + 1) # Must be a whole number
    W_out = int((W_in - W_k)/stride + 1) # Must be a whole number
    X_new = np.zeros((N, C_in, H_k*W_k, H_out*W_out))
    for i in range(H_out*W_out):
        row = int((i//W_out)*stride)
        col = int((i - row/stride*W_out)*stride)
        X_new[:, :, :, i] = X[:, :, row:row+H_k, col:col+W_k].reshape(N, C_in, H_k*W_k)
    return X_new

def im2col_naive(X, K, stride):
    N, C_in, H_in, W_in = X.shape
    C_out, _, H_k, W_k = K.shape
    H_out = int((H_in - H_k)/stride + 1) # Must be a whole number
    W_out = int((W_in - W_k)/stride + 1) # Must be a whole number
    
    rows = ((np.arange(H_out*W_out)//W_out)*stride).astype(int)
    cols = ((np.arange(H_out*W_out) - rows*W_out/stride)*stride).astype(int)
    
    X_new = [X[:,:,r:r+H_k,c:c+W_k].reshape(N, C_in, -1) for r,c in zip(rows, cols)]
    X_new = np.transpose(np.array(X_new), axes=[1,2,3,0])
    return X_new

def im2col_fast(X, K, stride):
    N, C_in, H_in, W_in = X.shape
    C_out, _, H_k, W_k = K.shape
    H_out = int((H_in - H_k)/stride + 1) # Must be a whole number
    W_out = int((W_in - W_k)/stride + 1) # Must be a whole number
    
    rows = ((np.arange(H_out*W_out)//W_out)*stride).astype(int)
    cols = ((np.arange(H_out*W_out) - rows*W_out/stride)*stride).astype(int)
   
    row_indices = np.repeat(rows[:, None] + np.arange(H_k), \
            repeats=H_k, axis=1)
    col_indices = np.tile(cols[:, None] + np.arange(W_k), \
            reps=[1,W_k])
    X_new = np.transpose(X[:,:,row_indices, col_indices], axes=[0,1,3,2])
    return X_new

def cnn_forward_im2col_fast(X, K, stride=1, pad_width=0):
    N, C_in, H_in, W_in = X.shape
    C_out, _, H_k, W_k = K.shape
    H_out = (H_in + 2*pad_width - H_k)//stride + 1
    W_out = (W_in + 2*pad_width - W_k)//stride + 1
    
    X_padded = zero_pad(X, pad_width=pad_width)
    # X_im2col shape (N, H_out*W_out, C_in*H_k*W_k)
    X_im2col = np.transpose(im2col_fast(X_padded, K, stride=stride), axes=[0,3,1,2]).\
            reshape(N*H_out*W_out, -1)
    # K_flat shape (C_out, C_in*H_k*W_k)
    K_flat = K.reshape(C_out, C_in*H_k*W_k)

    X_out = np.transpose((np.matmul(X_im2col, K_flat.T)).reshape(N, H_out, W_out, C_out), axes=[0,3,1,2])
    return X_out

def cnn_forward_im2col_naive(X, K, stride=1, pad_width=0):
    N, C_in, H_in, W_in = X.shape
    C_out, _, H_k, W_k = K.shape
    H_out = (H_in + 2*pad_width - H_k)//stride + 1
    W_out = (W_in + 2*pad_width - W_k)//stride + 1
    
    X_padded = zero_pad(X, pad_width=pad_width)
    # X_im2col shape (N, H_out*W_out, C_in*H_k*W_k)
    X_im2col = np.transpose(im2col_naive(X_padded, K, stride=stride), axes=[0,3,1,2]).\
            reshape(N*H_out*W_out, -1)
    # K_flat shape (C_out, C_in*H_k*W_k)
    K_flat = K.reshape(C_out, C_in*H_k*W_k)

    X_out = np.transpose((np.matmul(X_im2col, K_flat.T)).reshape(N, H_out, W_out, C_out), axes=[0,3,1,2])
    return X_out

