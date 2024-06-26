# --
# convex optimization - convex conjugate, infimal convolution and proximal map

import numpy as np
import matplotlib.pyplot as plt


def plot_function(x, f, name='no_name', xlim=None, ylim=None):
  """
  simple plot of function
  """
  plt.figure()
  plt.plot(x, f)
  plt.axis('square')
  plt.grid()
  plt.xlim(xlim)
  plt.ylim(ylim)
  #plt.show()
  plt.savefig(name + '.png', dpi=150)


def main():
  """
  main file for subdiffs
  """

  # x vector
  y = np.linspace(0.0001, 5, 1000)

  # 2.2 conj of e^x
  f2 = y * (np.log(y) - 1)

  f2[0] = 1e6
  


  # with cases
  # for i, xi in enumerate(x):

  #   if xi < 1:
  #     f3[i] = -xi + 1

  #   elif xi >= 1:
  #     f3[i] = (xi - 1) ** 2

  # plot of functions
  plot_function(y, f2, name='f2', xlim=(-2, 5), ylim=(min(f2) - 1, max(f2[1:])))
  #plot_function(x, f2, name='f2')
  #plot_function(x, f3, name='f3')



if __name__ == '__main__':
  main()
