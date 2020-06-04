import numpy as np
import scipy.sparse as sp

import imageio

import matplotlib.pyplot as plt


def calc_energy(u, u_0, D, lam):
    """
    calculate the energy of the minimization task
    """

    # error of reconstruction
    u_e = u - u_0

    # return energy
    return lam * np.linalg.norm(D @ u, ord=1) + 0.5 * (u_e.T @ u_e)


def get_subgradient(u, u_0, D, lam):
    """
    get the subgradient of the minimization function
    """

    # first subgradient
    f1 = lam * (np.sign(D @ u) @ D)

    # second subgradient
    f2 = u - u_0

    return f1 + f2


def subgradient_descent(u, u_0, D, lam, max_iter, t=None):
    """
    subgradient descent algorithm
    t is the step size, if None, the dynamic step size is used
    """

    u = u.copy()

    # init
    energy = np.zeros((max_iter,), dtype=np.float32)

    # print some infos:
    print("\n--subgradient descent algoritm")

    # over all iterations
    for k in range(max_iter):

        # get the subgradient
        g = get_subgradient(u, u_0, D, lam)

        # step size
        if t is None:

            # use dynamic step size
            t = 1 / (np.linalg.norm(g, ord=2) * np.sqrt(k + 1))

        # update image
        u = u - t * g

        # calculate energy
        energy[k] = calc_energy(u, u_0, D, lam)

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
  if (k % 10) == 0 or k == 0 or k==max_iter-1:

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


def plot_result(u, u_0, target):
    """
    plot result
    """

    # shape of things
    m, n = target.shape

    # plots
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(u_0.reshape(m, n), cmap='gray')
    ax[1].imshow(u.reshape(m, n), cmap='gray')
    ax[2].imshow(target.reshape(m, n), cmap='gray')


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

    # init u
    #u = u_0.copy()
    u = np.random.randn(*target.shape).astype(target.dtype).ravel()



    # --
    # params

    # max iterations
    max_iter = 1000

    # step size
    t_sgd = 0.01

    # lambda
    lams = [0.01, 0.1, 1.0]

    # choose lambda for testing
    lam = lams[0]
    #lam = 0.04

    # subgradient descent
    u, energy_sgd = subgradient_descent(u, u_0, D, lam, max_iter, t_sgd)


    # print end results
    print("\n--End results:")
    print("sgd:\t energy=[{:.4f}]".format(energy_sgd[-1]))

    # plot optimmization
    plot_result(u, u_0, target)

    # some plots
    #plot_diff(u_0, D_u)

    plt.show()