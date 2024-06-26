# --
# convex optimization - subdifferentials

import numpy as np
import matplotlib.pyplot as plt


def plot_function(x, f, name='no_name'):
  """
  simple plot of function
  """
  plt.figure()
  plt.plot(x, f)
  plt.axis('square')
  plt.grid()
  plt.xlim([-5, 5])
  plt.ylim([-2, 5])
  #plt.show()
  plt.savefig(name + '.png', dpi=150)


def main():
  """
  main file for subdiffs
  """

  # x vector
  x = np.linspace(-5, 5, 1000)

  # max function 
  f1 = np.maximum(x, np.zeros(len(x)))
  
  # abs function 
  f2 = np.abs(x)

  # weird function
  f3 = np.zeros(len(x))

  # with cases
  for i, xi in enumerate(x):

    if xi < 1:
      f3[i] = -xi + 1

    elif xi >= 1:
      f3[i] = (xi - 1) ** 2

  # plot of functions
  #plot_function(x, f1, name='f1')
  #plot_function(x, f2, name='f2')
  #plot_function(x, f3, name='f3')



if __name__ == '__main__':
  main()
