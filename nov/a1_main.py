import numpy as np
import matplotlib.pyplot as plt


def get_brackets(x0, s, k = 2):

  # first step
  a = x0
  b = x0 + s

  # check if downhill direction
  if fx(b) > fx(a):

    # wrong direction
    print('no minima in this direction')
    return (0, 0)

  # bracket algorithm
  stop_cond = False

  while not stop_cond:

    # step
    c = b + s

    # brackets found
    if fx(c) > fx(b):

      # update b
      b = c

      # stop
      stop_cond = True

    # brackets not found
    else:

      # step size is too high
      if s > 100:

        # no brackets exists
        stop_cond = True
        print('no brackets found')

      # update boundary
      a = b
      b = c

      # update step size
      s = s * k;

  return (a, b)


# --
# Main function
if __name__ == '__main__':

  # objective function
  fx = lambda x: np.sin(x) - np.sin(10/3 * x)

  # x space for function
  x = np.linspace(0, 5, 100)

  # --
  # line search

  # starting point
  x0 = 0.0

  # step size
  s = 0.01

  # step size multiplication factor
  k = 2

  # get the brackets
  a, b = get_brackets(x0, s, k)

  # sectioning
  # TODO


  # plot
  plt.figure(1)
  plt.plot(x, fx(x))

  # limit for brackets
  plt.xlim((a, b))

  plt.grid()
  plt.show()