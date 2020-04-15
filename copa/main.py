import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt

# load the data
data = np.load('./data-ass.npz')
# compressed image
x_0 = data['x_0'].astype(np.float32)
# target image
b = data['b']
# convolution kernels
k = data['k']

# define the shapes
m, n = b.shape
# the convolution kernel size
kernel_size = k.shape[-1]
# the number of kernels
N = len(k)

# choose initial coefficients
x = 1e-1*np.random.randn(N, (m+kernel_size-1), (n+kernel_size-1)).astype(np.float32)

# sparse convolution matrices
N, kernel_size, _ = k.shape
yy, xx = np.mgrid[:kernel_size,:kernel_size]
K = [None, ] * N
for l in range(N):
    # construct a full convolution matrix
    diag = yy.ravel()*x.shape[2] + xx.ravel()
    data = np.tile(k[l][::-1,::-1].reshape(-1, 1), (1, x[l].size))
    K_full = sp.diags(data, diag, (x[l].size, x[l].size), format='csr')
    # remove the boundary values
    mask = np.zeros_like(x[l], dtype=np.bool)
    mask[0:-kernel_size+1,0:-kernel_size+1] = 1
    K[l] = K_full[mask.ravel(),:]

# to speed up computation
x = x.reshape(N, -1)

# compute the initial estimate
x_hat = x_0 + sum([(K[i] @ x[i]).reshape(x_0.shape) for i in range(N)])

# compare the intermediate results
fig, ax = plt.subplots(1,3,sharex=True,sharey=True)
ax[0].imshow(x_0, cmap='gray')
ax[0].set_title(r'$x_0$')
ax[1].imshow(x_hat, cmap='gray')
ax[1].set_title(r'$\hat{x}$')
ax[2].imshow(b, cmap='gray')
ax[2].set_title(r'$b$')
plt.show()

# implement subgradient descent
def subgradient_descent(x, x_0, K, b, alpha, max_iter):
    energy = np.zeros((max_iter,), dtype=np.float32)
    ssd = np.zeros((max_iter,), dtype=np.float32)
    sparsity = np.zeros((max_iter,), dtype=np.float32)

    x = x.copy()

    # TODO:
    for k in range(max_iter):
       break

    return x, energy, ssd, sparsity

# implement the provided algorithm
def provided_algorithm(x, x_0, K, b, alpha, max_iter):
    energy = np.zeros((max_iter,), dtype=np.float32)
    ssd = np.zeros((max_iter,), dtype=np.float32)
    sparsity = np.zeros((max_iter,), dtype=np.float32)

    x = x.copy()
    # TODO:
    for k in range(max_iter):
        break

    return x, energy, ssd, sparsity

# TODO: choose properly
max_iter = 1
alpha = np.nan

# compute the reconstruction using subgradient descent
x_sgd, energy_sgd, ssd_sgd, sparsity_sgd = subgradient_descent(x, x_0, K, b.ravel(), alpha, max_iter)
x_hat_sgd = x_0 + sum([(K[i] @ x_sgd[i]).reshape(x_0.shape) for i in range(N)])

# compute the reconstruction using the provided algorithm
x_p, energy_p, ssd_p, sparsity_p = provided_algorithm(x, x_0, K, b.ravel(), alpha, max_iter)
x_hat_p = x_0 + sum([(K[i] @ x_p[i]).reshape(x_0.shape) for i in range(N)])

# plotting
# TODO: 
#   - plot the resulting reconstructions
#   - loglog plot of the energy, SSD and sparsity
