import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt


def calc_energy_primal(x, b, a_hat, D, lam):
  """
  calculate energy of the primal problem
  """

  # error of reconstruction
  x_e = A(x, a_hat) - b.ravel()

  # return energy
  return lam * np.linalg.norm(D @ x, ord=1) + 0.5 * (x_e.T @ x_e)


def unreg_deblur(a_full, b):
  """
  unregularized image deblurring
  """

  # shape of things
  M, N = b.shape

  # get the Fourier coefficients of the blur kernel
  a_hat = np.fft.fft2(a_full).ravel()
  b_hat = np.fft.fft2(b).ravel()

  # efficient diagonal matrix
  a_diag = 1 / a_hat

  # reconstruct
  x = np.fft.ifft2((a_diag * b_hat).reshape(M, N)).real

  return x


def prox_map_implicit():
  """
  proximal map of implicit solution
  """


def pdhg_algorithm(x, y, b, a_hat, D, lam, max_iter, tau, sigm, algo='explic'):
  """
  PDHG algorithm all possible solutions
  algo='explic':	explicit steps on the data fidelity term
  algo='implic':	implicit proximal steps on the data fidelity term
  algo='dual':		dualization of the quadratic data fidelity term
  """

  x = x.copy()
  y = y.copy()

  # init energy
  energy = np.zeros((max_iter,), dtype=np.float32)

  # print some infos:
  print("\n--pdhg algorithm: {}".format(algo))

  # over all iterations
  for k in range(max_iter):

    # explicit solution updates
    if algo == 'explic':

      x_pre = x
      x = x - tau * (D.T @ y + A_T(A(x, a_hat) - b.ravel(), a_hat))
      y = np.clip(y + sigm * D @ (2 * x - x_pre), -lam, lam)


    # implicit solution
    elif algo == 'implic':

      x_pre = x
      #x = 
      y = np.clip(y + sigm * D @ (2 * x - x_pre), -lam, lam)


    # dual solution
    else:

      # TODO:
      x = x
      y = y

    # calculate energy
    energy[k] = calc_energy_primal(x, b, a_hat, D, lam)

    # print some info
    print_iteration_info(k, energy[k], max_iter)

  return x, energy


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


def A(x, a_hat):
  '''
  implementation of the blur operator

  Parameters:
  x (np.ndarray): input image of shape (m*n,)

  Return: 
  np.ndarray: blurred image of shape (m*n,)
  '''
  x = x.reshape(m,n)
  a_hat = a_hat.reshape(m,n)
  # transform into fourier space
  x_f = np.fft.fft2(x)
  # convolve
  x_blur = a_hat * x_f
  # transfer back and take only the real part (imaginary part is close to zero)
  return np.fft.ifft2(x_blur).real.ravel()


def A_T(x, a_hat):
  """
  transpose blurring operator
  """

  return A(x, np.conj(a_hat))


def print_iteration_info(k, energy, max_iter):
  """
  print text info in each iteration
  """

  # print each 10-th time
  if (k % 10) == 9 or k == 0 or k==max_iter-1:

    # print info
    print("it: {} with energy=[{:.4f}] ".format(k + 1, energy))


def plot_result(a, b, name='img'):
  """
  plot the result
  """

  fig, ax = plt.subplots(1, 2)
  ax[0].imshow(a.reshape(m,n), cmap='gray')
  ax[1].imshow(b.reshape(m,n), cmap='gray')

  plt.savefig('./' + name + '.png', dpi=150)


def plot_end_result(b, x_hat, metrics, labels_metrics, labels_algo, name='end_result'):
  """
  plot the end result
  """

  # setup figure
  fig = plt.figure(figsize=(12, 8))

  # create a grid
  n_rows, n_cols = 2, 2
  gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)

  # titles
  t_list = [r'$b$', r'$x$']

  # vars
  x_list = [b, x_hat]
  pos = [(0, 0), (0, 1)]

  # plot images
  for t, x, p in zip(t_list, x_list, pos):

    # plot
    ax = fig.add_subplot(gs[p])
    ax.imshow(x.reshape(b.shape), cmap='gray')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(t)

  # plot metrics
  for i, metric in enumerate(metrics):

    # get axis
    ax = fig.add_subplot(gs[1, 0:])

    # plot all metric curves
    for m, l in zip(metric, labels_algo):
      ax.plot(m, label=l)

    # log scale
    ax.set_yscale('log')

    # set some labels
    ax.set_ylabel(labels_metrics[i])
    if i == len(metrics)-1:
      ax.set_xlabel('Iterations')

    ax.set_title("Primal Energies")
    ax.legend()
    ax.grid()

  plt.savefig('./' + name + '.png', dpi=150)


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

  # a wide-hat
  a_hat = np.fft.fft2(a_full).ravel()


  # init deblurred image randomly
  x = np.random.randn(m * n).astype(b.dtype)
  y = np.random.randn(m * n * 2).astype(b.dtype)

  # differential operator	
  D = get_D(m, n)

  # max iterations
  max_iter = 500

  # step size
  tau, sigm = 1, 1

  # lambda
  lam = 0.002

  # all algorithms for the pdhg
  algos = ['explic', 'implic', 'dual']

  # choose algo
  algo = algos[0]


  # unregularized image deblurring
  x_unreg = unreg_deblur(a_full, b)

  # pdhg algorithm
  x_pdhg1, energy_primal = pdhg_algorithm(x, y, b, a_hat, D, lam, max_iter, tau, sigm, algo=algo)


  # test blurring matrix
  x = A(b, a_hat)
  y = A_T(b, a_hat)
  z = A_T(A(b, a_hat), a_hat)


  # collect metrics
  metrics = [[energy_primal]]

  # labels of metrics
  labels_metrics, labels_algo = ['Energy'], ['primal']

  # param string for plots
  param_str = '_algo-{}_it-{}_lam-{}_tau-{}_sigm-{}'.format(algo, max_iter, str(lam).replace('.', 'p'), str(tau).replace('.', 'p'), str(sigm).replace('.', 'p'))

  # --
  # plots and print

  # end result
  plot_end_result(b, x_pdhg1, metrics, labels_metrics, labels_algo, name='end_result' + param_str)

  # plot result
  #plot_result(b, x_unreg)
  #plot_result(b, x)
  #plot_result(b, y)
  #plot_result(b, z)
  #plot_result(b, x_pdhg1, name='pdhg' + param_str)


  plt.show()