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
  k = [40, 80, 120, 160]
  A = [1, 0.5, 0.25, 1]

  x = np.dot(A, np.cos(2 * np.pi / N  * np.outer(k, np.arange(N)))) 

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

  print('inverse fft is equal: ', np.allclose(x, xi))

  # --
  # print fourier transformed signal

  plt.figure()
  plt.plot(Y)
  plt.plot(1 / N * np.abs(np.fft.fft(x)))
  plt.show()



  # --
  # cepstrum
  cep = cepstrum(x, N)


  # plot somethingthing
  # plt.figure(1)
  # plt.plot(cep)
  # plt.show()


  # --
  # ACF

  #x = np.random.normal(0, 1, N)

  r = acf(x)

  r2 = np.correlate(x, x, mode='full') / (2 * N + 1)

  print('correlation is equal: ', np.allclose(r[:-1], r2))


  plt.figure(2)
  plt.plot(x, label='x')
  plt.plot(r, label='r')
  plt.plot(r2, label='r2')
  plt.legend()
  plt.show()

