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

  # get the Fourier coefficients of the blur kernel
  a_hat = np.fft.fft2(a_full).ravel()
  b_hat = np.fft.fft2(b).ravel()

  # efficiently invert diagonal matrix
  a_diag_inv = 1 / a_hat

  # reconstruct
  x = np.fft.ifft2((a_diag_inv * b_hat).reshape(b.shape)).real

  return x


def prox_map_implicit(x, b, a_hat, tau):
  """
  proximal map of implicit solution
  """

  return F_T( (1 / (1 + tau * a_hat * np.conj(a_hat))) * (F(x) + tau * np.conj(a_hat) * F(b)) )


def prox_map_quadratic(z, sigm):
  """
  proximal map of quadratic solution
  """

  return z / (1 + sigm)


def pdhg_algorithm(x, y, z, b, a_hat, D, lam, max_iter, tau, sigm, algo='explic', dynamic_step=False):
  """
  PDHG algorithm all possible solutions
  algo='expl':	explicit steps on the data fidelity term
  algo='impl':	implicit proximal steps on the data fidelity term
  algo='quad':		dualization of the quadratic data fidelity term
  """

  x = x.copy()
  y = y.copy()
  z = z.copy()

  # init energy
  energy = np.zeros((max_iter,), dtype=np.float32)

  # print some infos:
  print("\n--pdhg algorithm: {}".format(algo))

  # over all iterations
  for k in range(max_iter):

    # explicit solution updates
    if algo == 'expl':

      # dynamic step size (experimental)
      if dynamic_step:
        tau = 1 / np.sqrt(k + 1)

      x_pre = x
      x = x - tau * (D.T @ y + A_T(A(x, a_hat) - b.ravel(), a_hat))
      y = np.clip(y + sigm * D @ (2 * x - x_pre), -lam, lam)


    # implicit solution
    elif algo == 'impl':

      # dynamic step size (experimental)
      if dynamic_step:
        tau = 1 / np.sqrt(k + 1)

      x_pre = x
      x = prox_map_implicit(x - tau * (D.T @ y), b, a_hat, tau)
      y = np.clip(y + sigm * D @ (2 * x - x_pre), -lam, lam)


    # dual of quadratic data fidelity solution
    else:

      # dynamic step size (experimental)
      if dynamic_step:
        tau = 1 / np.sqrt(k + 1)

      x_pre = x
      x = x - tau * (D.T @ y + A_T(z, a_hat))
      y = np.clip(y + sigm * D @ (2 * x - x_pre), -lam, lam)
      z = prox_map_quadratic(z + sigm * (A(2 * x - x_pre, a_hat) - b.ravel()), sigm)

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


def F(x):
  """
  2D fft transform
  """

  return np.fft.fft2(x.reshape(m, n)).ravel()


def F_T(x):
  """
  inverse 2D fft transform
  """

  return np.fft.ifft2(x.reshape(m, n)).real.ravel()


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

  fig, ax = plt.subplots(1, 2, figsize=(6, 3))
  ax[0].imshow(a.reshape(m,n), cmap='gray'), ax[0].set_title(r"$b$")
  ax[1].imshow(b.reshape(m,n), cmap='gray'), ax[1].set_title(r"$x_{D}$")
  plt.tight_layout()

  plt.savefig('./' + name + '.png', dpi=150)


def plot_end_result(imgs, imgs_titles, metrics, labels_metrics, labels_algo, name='end_result'):
  """
  plot the end result
  """

  # setup figure
  fig = plt.figure(figsize=(12, 8))

  # create a grid
  n_rows, n_cols = 1 + len(metrics), len(imgs)
  gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)

  # pos
  pos = [(0, 0), (0, 1), (0, 2), (0, 3)]

  # plot images
  for t, x, p in zip(imgs_titles, imgs, pos):

    # plot
    ax = fig.add_subplot(gs[p])
    ax.imshow(x.reshape(m, n), cmap='gray')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(t)

  # plot metrics
  for i, metric in enumerate(metrics):

    # get axis
    ax = fig.add_subplot(gs[1, 0:])

    # plot all metric curves
    for met, l in zip(metric, labels_algo):
      ax.plot(met, label=l)

    # log scale
    ax.set_yscale('log')
    ax.set_xscale('log')

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

  # unregularized image deblurring
  x_unreg = unreg_deblur(a_full, b)

  # init variables with zeros
  x = np.zeros(m * n).astype(b.dtype)
  y = np.zeros(2 * m * n).astype(b.dtype)
  z = np.zeros(m * n).astype(b.dtype)

  # differential operator	
  D = get_D(m, n)

  # max iterations
  max_iter = 200

  # step sizes (tau, sigm)
  step_sizes = [(1 / np.sqrt(8), 1 / np.sqrt(8)), (1 / np.sqrt(8), 1 / np.sqrt(8)), (1 / np.sqrt(8), 1 / np.sqrt(8))]
  #step_sizes = [(1, 1), (1, 1), (1, 1)]
  #step_sizes = [(1, 1/8), (1, 1/8), (1, 1/8)]
  #step_sizes = [(1/16, 0.5), (1/16, 0.5), (0.5, 2)]
  #step_sizes = [(0.5, 2), (0.5, 2), (0.5, 2)]
  #step_sizes = [(1, 2), (2, 2), (1, 2)]
  #step_sizes = [(1/8, 1/8), (1/8, 1/8), (1/8, 1/8)]
  #step_sizes = [(1/16, 1/16), (1/16, 1/16), (1/16, 1/16)]
  #step_sizes = [(1/100, 1), (1/100, 1), (1/100, 1)]

  # test number for saving plot
  test_num = 10

  # lambda
  lam = 0.001

  # all algorithms for the pdhg
  algos = ['expl', 'impl', 'quad']

  # init container vars
  imgs, energy_list,  = [b], []


  # run through all pdhg solutions
  for algo, step_size in zip(algos, step_sizes):

    # pdhg algorithm
    x_pdhg, energy_primal = pdhg_algorithm(x, y, z, b, a_hat, D, lam, max_iter, tau=step_size[0], sigm=step_size[1], algo=algo, dynamic_step=False)

    # update lists
    imgs.append(x_pdhg), energy_list.append(energy_primal)


  # collect metrics
  metrics = [energy_list]

  # labels of metrics
  labels_metrics, labels_algo = ['Energy'], [r'{}: $\tau = {:.4f}$, $\sigma = {:.4f}$'.format(algo, step[0], step[1]) for algo, step in zip(algos, step_sizes)]

  # param string for plots
  param_str = '_it-{}_lam-{}_num-{}'.format(max_iter, str(lam).replace('.', 'p'), test_num)

  # Lipschitz stuff
  print("D: max row: ", np.max(np.sum(np.abs(D), axis=0)))
  print("D: max col: ", np.max(np.sum(np.abs(D), axis=1)))
  print("a_hat l2:: ", np.max(a_hat, axis=0))


  # --
  # plots and print

  # titles
  imgs_titles = [r'$b$'] + [r'$x_{{{}}}$'.format(s) for s in algos]

  # end result
  plot_end_result(imgs, imgs_titles, metrics, labels_metrics, labels_algo, name='end_result' + param_str)

  # plot unreg result
  #plot_result(b, x_unreg, name='unreg')

  plt.show()