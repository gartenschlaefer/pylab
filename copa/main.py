import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt


def reconstruct_img(x, x_0, K):
    """
    reconstruct image from observation image x0, Kernel image K and coeffs x
    """
    return x_0 + sum([ (K[i] @ x[i]).reshape(x_0.shape) for i in range(len(K)) ])


def calc_energy(x, x_0, K, b, alpha):
    """
    calculate the energy of the minimization task
    """

    # calculate reconstruction [n1 x n2]
    x_hat = reconstruct_img(x, x_0, K)

    # error of reconstruction
    x_e = x_hat.ravel() - b.ravel()

    # energy
    E = np.sum([ np.linalg.norm(x[i], ord=1) + 0.5 * (x_e.T @ x_e) for i in range(len(K)) ])

    return E, x_hat


def subgradient_descent(x, x_0, K, b, alpha, max_iter):
    """
    subgradient descent algorithm
    """

    # init
    energy = np.zeros((max_iter,), dtype=np.float32)
    ssd = np.zeros((max_iter,), dtype=np.float32)
    sparsity = np.zeros((max_iter,), dtype=np.float32)

    x = x.copy()

    # TODO: implementation
    for k in range(max_iter):
       break

    return x, energy, ssd, sparsity


def provided_algorithm(x, x_0, K, b, alpha, max_iter):
    """
    provided algorithm from the assignment sheet
    """

    # init
    energy = np.zeros((max_iter,), dtype=np.float32)
    ssd = np.zeros((max_iter,), dtype=np.float32)
    sparsity = np.zeros((max_iter,), dtype=np.float32)

    x = x.copy()

    # Lipschitz constant
    L = 10

    # get shapes
    n = len(b)
    n1, n2 = x_0.shape
    k, m = x.shape

    # print some infos:
    print("\n--provided algorithm--")
    print("image: [n1 x n2] = [{} x {}], n=[{}], m=[{}], k=[{}]".format(n1, n2, n, m, k))
    print("params: iterations:{}, alpha=[{}], L=[{}]".format(max_iter, alpha, L))

    # shape of things
    # b:    [n]         - image
    # x_0:  [n1 x n2]   - observation
    # x:    [k x m]     - coeffs of kernel
    # K:    [k x n x m] - conv img with kernel

    # iterations
    for it in range(max_iter):

        # reconstruction error [n]
        x_e = reconstruct_img(x, x_0, K).ravel() - b

        # for each kernel
        for i in range(len(K)):

            # x bar [m]
            x_bar = x[i] - 1 / L * K[i].T @ x_e

            # update coeffs x [k x m]
            x[i] = np.maximum( np.abs(x_bar) - alpha / L, np.zeros(x_bar.shape) ) * (np.sign(x_bar) + (x_bar == 0))

        # calculate energy
        energy[it], x_hat = calc_energy(x, x_0, K, b, alpha)

        # print iteration
        #plot_compare_results(x_0, x_hat, b.reshape(n1, n2))
        print("iteration: {} with energy=[{}] ".format(it, energy[it]))

    return x, energy, ssd, sparsity


def plot_compare_results(x_0, x_hat, b):
  """
  compare the intermediate results
  """
  fig, ax = plt.subplots(1,3,sharex=True,sharey=True, figsize=(9, 7))
  ax[0].imshow(x_0, cmap='gray')
  ax[0].set_title(r'$x_0$')
  ax[1].imshow(x_hat, cmap='gray')
  ax[1].set_title(r'$\hat{x}$')
  ax[2].imshow(b, cmap='gray')
  ax[2].set_title(r'$b$')
  plt.show()


if __name__ == '__main__':
  """
  main function
  """

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
  x_hat = reconstruct_img(x, x_0, K)
  x_hat = x_0 + sum([(K[i] @ x[i]).reshape(x_0.shape) for i in range(N)])

  # plot results
  #plot_compare_results(x_0, x_hat, b)


  # TODO: choose properly
  max_iter = 10
  alpha = 0.1

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

  # plot energy string
  print("\nsgd: energy={}, \nssd: energy={}".format(energy_sgd, energy_p))

  # plot end results
  plot_compare_results(x_0, x_hat_p, b)


  plt.show()
