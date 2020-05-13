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


def calc_sparsity(x, eps=1e-3):
  """
  calculate the sparsity of a matrix, return percentage value
  """
  return np.sum(np.abs(x) >= eps) / np.product(x.shape)


def calc_alpha(x, perc=0.1):
  """
  calculate alpha so that 10% of x are non-zero
  """

  # init alpha
  alpha = 1.0

  # step size
  s = 0.001

  # stopping criterion
  stop_criterion = False

  while not stop_criterion:

    # sparsity condition
    x_s = alpha * x

    # caclulate sparsity
    S = calc_sparsity(x_s, eps=1e-3)

    print("S: {} \t alpha: {}".format(S, alpha))

    # stopping condition
    if S <= perc:

      stop_criterion = True
      return alpha

    # update alpha
    alpha -= s 

  return alpha


def get_subgradient(x):
    """
    get the subgradient of the minimization function
    """

    # TODO: impelentation
    return 1


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

      # get subgradient
      g = get_subgradient(x)

      # select step size

      t = 0.1

      # Polyak
      #t = f / np.linalg.norm(g,2)**2

      # Dynamic
      #t = 1/(np.linalg.norm(g,2)*np.sqrt(iter+1))

      # update params
      x = x - t * g

      # calculate energy
      energy[k], _ = calc_energy(x, x_0, K, b, alpha)


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

    # Lipschitz constant (choose high enough)
    L = 10

    # get shapes
    n = len(b)
    n1, n2 = x_0.shape
    N, m = x.shape

    # print some infos:
    print("\n--provided algorithm--")
    print("image: [n1 x n2] = [{} x {}], n=[{}], m=[{}], k=[{}]".format(n1, n2, n, m, N))
    print("params: iterations:{}, alpha=[{}], L=[{}]".format(max_iter, alpha, L))

    # shape of things
    # b:    [n]         - image (flattened): n = n1 * n2
    # x_0:  [n1 x n2]   - observation
    # x:    [N x m]     - coeffs of kernel
    # K:    [N x n x m] - conv img with kernel

    # iterations
    for k in range(max_iter):

        # reconstruction error: [n]
        x_e = reconstruct_img(x, x_0, K).ravel() - b

        # for each kernel
        for i in range(len(K)):

            # x_bar: [m]
            x_bar = x[i] - 1 / L * K[i].T @ x_e

            # update coeffs x: [N x m]
            x[i] = np.maximum( np.abs(x_bar) - alpha / L, np.zeros(x_bar.shape) ) * (np.sign(x_bar) + (x_bar == 0))

        # calculate energy
        energy[k], x_hat = calc_energy(x, x_0, K, b, alpha)

        # calculate sparsity
        sparsity[k] = calc_sparsity(x, eps=1e-3)

        # print iteration
        #plot_compare_results(x_0, x_hat, b.reshape(n1, n2))
        print("iteration: {} with energy=[{}] ".format(k, energy[k]))

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
  #plt.show()


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
  x_hat_init = reconstruct_img(x, x_0, K)

  # determine alpha
  #alpha = calc_alpha(x)
  #print("alpha: ", alpha)

  # TODO: choose properly
  max_iter = 5
  alpha = 0.006

  # compute the reconstruction using subgradient descent
  x_sgd, energy_sgd, ssd_sgd, sparsity_sgd = subgradient_descent(x, x_0, K, b.ravel(), alpha, max_iter)
  x_hat_sgd = reconstruct_img(x_sgd, x_0, K)

  # compute the reconstruction using the provided algorithm
  x_p, energy_p, ssd_p, sparsity_p = provided_algorithm(x, x_0, K, b.ravel(), alpha, max_iter)
  x_hat_p = reconstruct_img(x_p, x_0, K)

  # plotting
  # TODO: 
  #   - plot the resulting reconstructions
  #   - loglog plot of the energy, SSD and sparsity

  # plot energy string
  print("\nsgd:\t energy={}, sparsity={} \nprovided:\t energy={}, sparsity={}".format(energy_sgd, sparsity_sgd, energy_p, sparsity_p))

  # plot end results
  plot_compare_results(x_0, x_hat_init, b)
  plot_compare_results(x_0, x_hat_sgd, b)
  plot_compare_results(x_0, x_hat_p, b)

  # plt.figure()
  # plt.plot(sparsity_p)
  # plt.show()


  plt.show()
