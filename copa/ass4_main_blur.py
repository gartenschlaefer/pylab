import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt


def calc_energy_primal(x, b, D, lam):
	"""
	calculate energy of the primal problem
	"""

	# TODO
	return None

	# error of reconstruction
	x_e = A(x) - b

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


def pdhg_algorithm(x, b, D, lam, max_iter, tau, algo='explic'):
	"""
	PDHG algorithm all possible solutions
	algo='explic':	explicit steps on the data fidelity term
	algo='implic':	implicit proximal steps on the data fidelity term
	algo='dual':		dualization of the quadratic data fidelity term
	"""

	x = x.copy()

	# init energy
	energy = np.zeros((max_iter,), dtype=np.float32)

	# print some infos:
	print("\n--pdhg algorithm: {}".format(algo))

	# over all iterations
	for k in range(max_iter):

		# explicit solution updates
		if algo == 'explic':

			#x = x - tau * (D.T @ y + A.)
			pass


		# calculate energy
		energy[k] = calc_energy_primal(x, b, D, lam)

		# print some info
		print_iteration_info(k, energy[k], max_iter)

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


def print_iteration_info(k, energy, max_iter):
  """
  print text info in each iteration
  """

  # print each 10-th time
  if (k % 10) == 9 or k == 0 or k==max_iter-1:

    # print info
    print("it: {} with energy=[{:.4f}] ".format(k + 1, energy))


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


	# init deblurred image randomly
	x = np.random.randn(m * n).astype(b.dtype)

	# differential operator	
	D = get_D(m, n)
	#A = get_A()

	# max iterations
	max_iter = 10

	# step size
	tau = 0.01

	# lambda
	lam = 0.1

	# unregularized image debluring
	x_unreg = unreg_deblur(a_full, b)

	x_pdhg1 = pdhg_algorithm(x, b, D, lam, max_iter, tau, algo='explic')


	# --
	# plots and print

	# shape
	print("image shape: ", b.shape)

	# plot result
	plot_result(b, x_unreg)


	plt.show()