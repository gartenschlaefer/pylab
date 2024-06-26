# --
# fourier transform lab

import numpy as np
import matplotlib.pyplot as plt

def my_dft(x, K=512):
  """
  simple fourier
  """

  N = len(x)
  F = np.exp(1j * 2 * np.pi / N * np.outer(np.arange(K), np.arange(N)))

  R = np.cos(2 * np.pi / N * np.outer(np.arange(K), np.arange(N)))
  I = np.sin(2 * np.pi / N * np.outer(np.arange(K), np.arange(N)))

  F_ = R + 1j * I

  #plot_image(R)
  #plot_image(I)

  #x_tilde = F_ @ x
  x_tilde = F @ x
  print("F: ", F.shape)
  print("x_tilde: ", x_tilde.shape)

  return x_tilde


def plot_image(x):
  """
  image plot
  """

  plt.figure()

  # create axis
  ax = plt.axes()
  ax.imshow(x)
  plt.show()


def plot_waveform(y, x):
  """
  plot waveform
  """

  plt.figure()

  # create axis
  ax = plt.axes()

  ax.plot(x, y)
  plt.show()


if __name__ == '__main__':
  """
  main
  """

  N = 128
  K = N
  n = np.arange(N)
  f = np.arange(K)
  a1 = 1
  a2 = 0
  f1 = N // 4
  f2 = 40

  # signal
  x = a1 * np.sin(2 * np.pi * f1 * n / N) + a2 * np.sin(2 * np.pi * f2 * n / N)

  plot_waveform(x, n)

  x_tilde = my_dft(x, K)

  x_tilde_abs = np.abs(x_tilde)
  x_tilde_phi = np.angle(x_tilde)
  print(x_tilde_phi)
  #stop
  plot_waveform(x_tilde_abs, f)
  plot_waveform(x_tilde_phi, f)