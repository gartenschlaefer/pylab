import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from mia import *

# --
# main
if __name__ == '__main__':

  # params
  fs = 44100

  # sample params
  N = 1024

  # windowing params
  ol = N // 2
  hop = N - ol

  # test signal
  k = 40
  A = 3
  x = A * np.cos(2 * np.pi * k / N * np.arange(N))

  # some vectors
  t = np.arange(0, N/fs, 1/fs)

  # transformation matrix
  H = np.exp(1j * 2 * np.pi / N * np.outer(np.arange(N), np.arange(N)))

  # transfored signal
  X = np.dot(x, H)

  xi = (np.dot(X, H) / N).real

  # log
  #Y = 20 * np.log10(2 / N * np.abs(X[0:512]))
  Y = 2 / N * np.abs(X[0:512])


  print('is equal: ', np.allclose(x, xi))

  cep = cepstrum(x, N)

  # plot somethingthing
  plt.figure(1)
  plt.plot(cep)
  plt.show()

