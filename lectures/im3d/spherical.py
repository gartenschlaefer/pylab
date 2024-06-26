# --
# spherical harmonic analysis

import numpy as np
import matplotlib.pyplot as plt


def plot_cardioid(theta, order=1):
  """
  plot a simple cardioid
  """
  r = (1 + np.cos(theta))**order 

  ax = plt.subplot(111, projection='polar')
  ax.plot(theta, r)
  plt.show()


def plot_polardir(alpha=45):
  """
  plot a polar direction
  """
  theta_s = alpha * np.pi / 180 * np.ones(2)
  r_s = np.array([0, 1])
  ax = plt.subplot(111, projection='polar')
  ax.plot(theta_s, r_s)
  plt.show()


def plot_sin_cos(theta):
  """
  sin * cos 
  """
  plt.figure()
  a, b = np.meshgrid(np.cos(theta), np.sin(theta))
  r = a * b
  plt.plot(r)
  a, b = np.meshgrid(np.sin(theta), np.sin(theta))
  r = a * b
  plt.plot(r)
  plt.show()


def main():
  """
  main function
  """

  # theta for polar plots
  theta = np.arange(0, 2 * np.pi, 0.01)

  # some plots
  #plot_polardir(alpha=45)

  plot_cardioid(theta, order=5)


  # plt.figure()
  # for n in range(1, 5):
  #   plt.plot(theta, np.cos(n * theta))
  # plt.show()










if __name__ == '__main__':
  main()