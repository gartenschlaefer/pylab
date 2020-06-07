import numpy as np
import scipy.sparse as sp

import imageio

import matplotlib.pyplot as plt


def calc_energy_primal(u, u_0, D, lam):
    """
    calculate energy of the primal problem
    """

    # error of reconstruction
    u_e = u - u_0

    # return energy
    return lam * np.linalg.norm(D @ u, ord=1) + 0.5 * (u_e.T @ u_e)


def calc_energy_dual(p, u_0, D, lam):
    """
    calculate energy of the dual problem
    """

    # dual reconstruction
    D_p = D.T @ p

    # return energy
    return -0.5 * (D_p.T @ D_p) + (D_p @ u_0)


def get_subgradient_primal(u, u_0, D, lam):
    """
    subgradient of the primal problem
    """

    # first subgradient
    f1 = lam * (np.sign(D @ u) @ D)

    # second subgradient
    f2 = u - u_0

    return f1 + f2


def get_subgradient_dual(p, u_0, D):
    """
    subgradient for the dual problem
    """

    # first subgradient
    f1 = -D.T @ p @ D.T
    #f1 = - D @ D.T @ p

    # second subgradient
    f2 = D @ u_0

    return f1 + f2


def prox_map_dual(p, g, t, lam):
    """
    proximal map for the dual problem
    """

    # compute update rule (ascent) x:[2n]
    x = p + t * g

    # return projection
    return x / np.maximum(np.ones(x.shape), np.abs(x))
    #return x / np.maximum(np.ones(x.shape) * lam, np.abs(x))
    #return x


def gradient_ascent(p, u_0, D, lam, max_iter, t=None):
    """
    gradient ascent algorithm
    t is the step size, if None, the dynamic step size is used
    """

    p = p.copy()

    # init
    energy_dual = np.zeros((max_iter,), dtype=np.float32)
    energy_prim = np.zeros((max_iter,), dtype=np.float32)

    # print some infos:
    print("\n--gradient ascent algorithm")

    # over all iterations
    for k in range(max_iter):

        # get the subgradient
        g = get_subgradient_dual(p, u_0, D)

        # dynamic step size
        if t is None:

            # use dynamic step size
            t = 1 / (np.linalg.norm(g, ord=2) * np.sqrt(k + 1))

        # proximal map
        p = prox_map_dual(p, g, t, lam)

        # calculate energy
        energy_dual[k] = calc_energy_dual(p, u_0, D, lam)
        energy_prim[k] = calc_energy_primal(u_0 - D.T @ p, u_0, D, lam)

        # print iteration info
        print_iteration_info(k, energy_dual[k], max_iter)

    # primal solution from dual problem
    u = u_0 - D.T @ p

    return u, energy_dual, energy_prim


def subgradient_descent(u, u_0, D, lam, max_iter, t=None):
    """
    subgradient descent algorithm
    t is the step size, if None, the dynamic step size is used
    """

    u = u.copy()

    # init
    energy = np.zeros((max_iter,), dtype=np.float32)

    # print some infos:
    print("\n--subgradient descent algorithm")

    # over all iterations
    for k in range(max_iter):

        # get the subgradient
        g = get_subgradient_primal(u, u_0, D, lam)

        # step size
        if t is None:

            # use dynamic step size
            t = 1 / (np.linalg.norm(g, ord=2) * np.sqrt(k + 1))

        # update image
        u = u - t * g

        # calculate energy
        energy[k] = calc_energy_primal(u, u_0, D, lam)

        # print iteration info
        print_iteration_info(k, energy[k], max_iter)

    return u, energy


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


def print_iteration_info(k, energy, max_iter):
  """
  print text info in each iteration
  """

  # print each 10-th time
  if (k % 10) == 9 or k == 0 or k==max_iter-1:

    # print info
    print("it: {} with energy=[{:.4f}]] ".format(k + 1, energy))


def plot_diff(u_0, D_u):
    """
    plot difference operator on target image
    """

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(u_0.reshape(m,n),cmap='gray')
    ax[1].imshow(D_u.reshape(2,m,n)[0],cmap='gray')
    ax[2].imshow(D_u.reshape(2,m,n)[1],cmap='gray')


def plot_result(u_prim, u_dual, u_0, target):
    """
    plot result
    """

    # shape of things
    m, n = target.shape

    # plots
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(u_0.reshape(m, n), cmap='gray')
    ax[1].imshow(u_prim.reshape(m, n), cmap='gray')
    ax[2].imshow(u_dual.reshape(m, n), cmap='gray')
    ax[3].imshow(target.reshape(m, n), cmap='gray')


def plot_end_result(u_prim, u_dual, u_0, target, metrics, labels_metrics, labels_algo, max_iter, lam, fn_coda=''):
  """
  plot the end result
  """

  # setup figure
  fig = plt.figure(figsize=(12, 8))

  # create a grid
  n_rows, n_cols = 2, 4
  gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)

  # titles
  t_list = [r'$u_0$', r'$\hat{u}_{target}$', r'$\hat{u}_{primal}$', r'$\hat{u}_{dual}$']

  # vars
  x_list = [u_0, target, u_prim, u_dual]
  pos = [(0, 0), (0, 1), (0, 2), (0, 3)]

  # plot images
  for t, x, p in zip(t_list, x_list, pos):

    # plot
    ax = fig.add_subplot(gs[p])
    ax.imshow(x.reshape(target.shape), cmap='gray')
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

    ax.set_title(r'Primal Energies with param: $\lambda = $' + str(lam))
    ax.legend()
    ax.grid()

  plt.savefig('./end_result_it-' + str(max_iter) + '_' + 'lam-' + str(lam).replace('.','p') + fn_coda + '.png', dpi=150)


if __name__ == '__main__':
    """
    main function
    """

    # load the image
    target = imageio.imread('orig.png').astype(np.float32)/255

    # get shape of things
    m, n = target.shape

    # create the noisy image
    u_0 = target + 0.05 * np.random.randn(*target.shape).astype(target.dtype)
    u_0 = u_0.ravel()

    # test the linear operator
    D = get_D(m, n)

    # apply linear operator
    D_u = D @ u_0

    # init primal and dual
    #u = u_0.copy()
    u = np.random.randn(n * m).astype(target.dtype).ravel()
    p = np.random.randn(2 * n * m).astype(target.dtype)



    # --
    # params

    # max iterations
    max_iter = 500
    #max_iter = 50

    # step size
    t_primal = 0.01
    t_dual = 0.01

    # lambda
    lams = [0.01, 0.1, 1.0]

    # choose lambda for testing
    lam = lams[1]
    #lam = 0.04

    # subgradient descent
    u_primal, energy_primal = subgradient_descent(u, u_0, D, lam, max_iter, t_primal)

    # gradient ascent of dual problem
    u_dual, energy_dual, energy_dual_primal = gradient_ascent(p, u_0, D, lam, max_iter, t_dual)


    # print end results
    print("\n--End results:")
    print("primal:\t energy=[{:.4f}]".format(energy_primal[-1]))
    print("dual:\t energy=[{:.4f}]".format(energy_dual[-1]))


    # --
    # some further plots

    # collect metrics
    metrics = [[energy_primal, energy_dual_primal]]
  
    # labels of metrics
    labels_metrics, labels_algo = ['Energy'], ['primal', 'dual']

    # end result
    plot_end_result(u_primal, u_dual, u_0, target, metrics, labels_metrics, labels_algo, max_iter, lam, fn_coda='_test')

    # plot optimization
    #plot_result(u_primal, u_dual, u_0, target)

    print("shapes: ", target.shape)
    print("max row: ", np.max(np.sum(np.abs(D), axis=0)))
    print("max col: ", np.max(np.sum(np.abs(D), axis=1)))

    # some plots
    #plot_diff(u_0, D_u)

    plt.show()