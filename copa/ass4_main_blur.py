import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt


def unreg_deblur(a_full, b):
    """
    unregularized image deblurring
    """

    # get the Fourier coefficients of the blur kernel
    a_tilde = np.fft.fft2(a_full)
    b_tilde = np.fft.fft2(b)

    print("a tilde: ", a_tilde.shape)
    print("b tilde: ", b_tilde.shape)


    x = np.fft.ifft2(np.linalg.inv(np.diag(np.diag(a_tilde))) @ np.fft.fft2(b))

    plot_result(np.abs(x), np.abs(a_tilde))

    print("x: ", x.shape)

    return x


# construct the linear difference operator
def get_D(M,N):
    '''
    approximation of the image gradient by forward finite differences

    Parameters:
    M (int): height of the image
    N (int): width of the image

    Return: 
    scipy.sparse.coo_matrix: Sparse matrix extracting the image gradients
    '''
    row = np.arange(0,M*N)
    dat = np.ones(M*N, dtype=np.float32)
    col = np.arange(0,M*N).reshape(M,N)
    col_xp = np.hstack([col[:,1:], col[:,-1:]])
    col_yp = np.vstack([col[1:,:], col[-1:,:]])
    
    FD1 = sp.coo_matrix((dat, (row, col_xp.flatten())), shape=(M*N, M*N))- \
          sp.coo_matrix((dat, (row, col.flatten())), shape=(M*N, M*N))

    FD2 = sp.coo_matrix((dat, (row, col_yp.flatten())), shape=(M*N, M*N))- \
          sp.coo_matrix((dat, (row, col.flatten())), shape=(M*N, M*N))
    
    FD = sp.vstack([FD1, FD2])
    
    return FD


def A(x):
    '''
    implementation of the blur operator

    Parameters:
    x (np.ndarray): input image of shape (m*n,)

    Return: 
    np.ndarray: blurred image of shape (m*n,)
    '''
    x = x.reshape(m,n)
    # transform into fourier space
    x_f = np.fft.fft2(x)
    # convolve
    x_blur = a_tilde * x_f
    # transfer back and take only the real part (imaginary part is close to zero)
    return np.fft.ifft2(x_blur).real.ravel()


def plot_result(a, b):
    """
    plot the result
    """

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(a, cmap='gray')
    ax[1].imshow(b.reshape(m,n), cmap='gray')


if __name__ == '__main__':
    """
    main file
    """

    # load the data
    data = np.load('data.npz')

    # kernel, blurred image
    a, b = data['a'], data['b']

    # get the shape
    m, n = b.shape

    # compute the Fourier transform (image space) of the convolution kernel 
    a_full = np.zeros((m,n), dtype=a.dtype)
    a_full[:a.shape[0],:a.shape[1]] = a

    # ensure that the response is centered
    for i in [0,1]:
        a_full = np.roll(a_full, -a.shape[i]//2, i)


    # unregularized image debluring
    x = unreg_deblur(a_full, b)


    # --
    # plots and print

    # shape
    print("image shape: ", b.shape)

    # plot result
    #plot_result(a, b)


    plt.show()