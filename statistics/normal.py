# --
# normal dist

import numpy as np
import matplotlib.pyplot as plt


# lambda function
f_normal = lambda x, mu, std : (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(1 / 2) * ((x - mu) / std)**2)

if __name__ == '__main__':
  """
  normal dist
  """

  # range
  N = 1000
  x = np.linspace(-5, 5, N)

  dx = x[1] - x[0]
  print(dx)

  # len
  print("N: ", N)

  # prob. density function
  p1 = f_normal(x, 0, 1)
  p2 = f_normal(x, 1, np.sqrt(0.2))
  p3 = f_normal(x, -2, 1.5)

  print("sum x: ", sum(x))
  e_p2 = sum(x * p2) * dx
  print("mu est: ", e_p2)

  # combined
  p = p1 * p2 * p3
  # p = p1 + p2 + p3

  # cumulative
  C = np.stack([np.concatenate((np.ones(i), np.zeros(N-i))) for i in range(1, N + 1)], axis=0)
  F_p2 = C @ p2 * dx
  F_p = C @ p * dx


  # plot
  plt.figure(), plt.plot(x, p2), plt.plot(x, F_p2), plt.show()
  plt.figure(), plt.plot(x, p1), plt.plot(x, p2), plt.plot(x, p3), plt.show()
  plt.figure(), plt.plot(x, p), plt.show()
  plt.figure(), plt.plot(x, F_p), plt.show()



  print("N: ", N)
  print("sum: ", np.sum(p))
