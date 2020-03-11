import matplotlib.pyplot as plt
import numpy as np

def line_between_points(a, b):
  """
  draws a line between two points in 2D
  """

  x = np.zeros(2)
  y = np.zeros(2)

  x[0] = a[0]
  y[0] = a[1]

  x[1] = b[0]
  y[1] = b[1]

  return x, y



def main():

  # convex sets

  # 1. prove conv
  e1 = np.array([1, 0])
  e2 = np.array([0, 1])

  #x = np.arange(0, M)
  #y = np.arange(0, N)

  # meshgrid
  #X, Y = np.meshgrid(x, y)

  # discrete points in a mesh
  x1, x2 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))

  x_conv = x1**2 + x2**2 - 1.0

  print("x1: ", x1.shape)
  #print("x1: ", x1)
  #print("x2: ", x2)

  #np.linalg.norm(a, ord=1, axis=0)

  #l1_norm = lambda x : np.sum(np.abs(x))

  print("line: ", line_between_points(e1, -e1))

  plt.figure()

  plt.contour(x1, x2, x_conv, [0])

  # plot lines between the points
  for a, b in [(e1, -e1), (e2, -e2), (e1, e2), (e1, -e2), (-e1, e2), (-e1, -e2)]:
    x, y = line_between_points(a, b)
    plt.plot(x, y, color='yellowgreen', linewidth=1.5)

  # plot points
  plt.scatter(e1[0], e1[1], label='e1')
  plt.scatter(e2[0], e2[1], label='e2')
  plt.scatter(-e1[0], -e1[1], label='-e1')
  plt.scatter(-e2[0], -e2[1], label='-e2')

  plt.axis('equal')

  #plt.plot(e2, -e1)
  #plt.plot(e2, -e2)

  #plt.scatter(x1, x2, label='set')

  plt.legend()
  plt.grid()
  plt.tight_layout()
  plt.show()


if __name__== "__main__":
  main()