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
  k = 40.22
  A = 3
  x = A * np.cos(2 * np.pi * k / N * np.arange(2.254 * N))

  # some vectors
  t = np.arange(0, N/fs, 1/fs)
  f = np.arange(0, fs/2, fs/N)

  # windowing
  w = np.hanning(N)

  # apply windows
  x_buff = np.multiply(w, buffer(x, N, ol))

  # transformation matrix
  H = np.exp(1j * 2 * np.pi / N * np.outer(np.arange(N), np.arange(N)))

  # transfored signal
  X = np.dot(x_buff, H)

  # log
  Y = 20 * np.log10(2 / N * np.abs(X[1][0:512]))

  # peaks
  #p = signal.find_peaks_cwt(X, np.arange(1, len(X)))
  p, _ = signal.find_peaks(Y, height=(-100, 100))
  print("---peaks at sample: ", p)

  # parabolic interpolation
  a, b, g, k_par = parabol_interp(Y, p[0])

  print("---parabol interpolation:")
  print('f_real: ', k * fs / N)
  print('f_peak: ', p * fs / N)
  print('f_est: ', k_par * fs / N)
  print('k_par: ', k_par)
  print("--")
  print("A_real: ", A)
  print('A_peak: ', np.power(10, Y[p[0]] / 20))
  print('A_est: ', np.power(10, b / 20))

  print("---phase derivation:")
  print('f_real: ', k * fs / N)
  print('f_peak: ', p * fs / N)
  print('f_est: ', inst_f(X, 1, p, hop, N, fs))



  # plot somethingthing
  plt.figure(12)
  plt.subplot(211)
  plt.plot(f, Y, label='cosine')
  plt.grid()
  plt.xlabel('frequency')
  plt.ylabel('magnitude')

  plt.subplot(212)
  plt.plot(f, np.angle(X[1][0:512]), label='cosine')
  plt.grid()
  plt.xlabel('frequency')
  plt.ylabel('angle')
  plt.legend()

  plt.show()



  # --
  # testing my buffer function
  #
  # a = np.concatenate( (np.zeros(512), np.ones(512), np.zeros(512) + 5, np.ones(512) + 5, np.zeros(100) + 10 ))
  # a_buff = buffer(a, N, OL)

  # plt.figure(111)
  # for wi in range(len(a_buff)):
  #   plt.plot(np.arange(N), a_buff[wi], label='window' + str(wi))
  # plt.xlabel('samples')
  # plt.ylabel('magnitude')
  # plt.legend()
  # plt.show()


  # --
  # testing window functions
  #
  # plt.figure(1)
  # plt.plot(np.arange(N), w, label='hanning')
  # plt.grid()
  # plt.xlabel('samples')
  # plt.ylabel('magnitude')
  # plt.show()


  # --
  # parabol testing
  #
  # print("a: ", a, " b: ", b, " g: ", g, " k: ", k_par)
  # n_par = np.linspace(-1, 1, 100)
  # y_par = a * np.power( n_par - g, 2) + b
  # y_values = np.array([ Y[p[0]-1], Y[p[0]], Y[p[0]+1]  ])
  #
  # # check parabol stuff
  # plt.figure(1)
  # plt.plot(np.linspace(-1, 1, 3), y_values, label='f')
  # plt.plot(n_par, y_par, label='parabel')
  # plt.grid()
  # plt.xlabel('frequency')
  # plt.ylabel('magnitude')
  # plt.legend()
  # #plt.show()