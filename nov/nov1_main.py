import numpy as np
import matplotlib.pyplot as plt


def color_fader(c1, c2, mix):
  """
  fades colors
  """
  import matplotlib as mpl

  # convert colors to rgb
  c1 = np.array(mpl.colors.to_rgb(c1))
  c2 = np.array(mpl.colors.to_rgb(c2))

  return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def get_brackets(x0, s, k = 2):
  """
  get brackets of minimizing function fx(x)
  """

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


def sectioning(a, b, sec_lim=1e-3):
  """
  sectioning of the brackets with the golden ration
  """
  gold = (1 + np.sqrt(5)) / 2

  # list of sections for print
  sec_list = np.empty((0, 2), float)

  while np.abs(b - a) > sec_lim:

    # make smaller section
    s_sec = (gold - 1) * (b - a)

    # new boundaries
    c = b - s_sec;
    d = a + s_sec;

    # update sections
    a = c
    b = d

    # update list
    sec_list = np.vstack((sec_list, (a, b)))

  return (a + b) / 2, sec_list


def plot_sectioning(x, sec_list):
  # plot
  plt.figure(1, figsize=(8, 4))

  # plot obj function
  plt.plot(x, fx(x))

  # colors mixing
  c1 = 'green'
  c2 = 'orange'
  c_mix = np.linspace(0, 1, len(sec_list))
  a_mix = np.linspace(0.5, 1, len(sec_list))

  # plot brackets
  for i, sec in enumerate(sec_list):
    c = color_fader(c1, c2, c_mix[i])
    plt.axvline(x=sec[0], color=c, dashes=(4, 2), alpha=a_mix[i], label='sec'+str(i+1))
    plt.axvline(x=sec[1], color=c, dashes=(4, 2), alpha=a_mix[i])

  # limit for brackets
  plt.ylim((-0.8, -0.2))
  plt.xlim((a, b))

  plt.ylabel('f(x)')
  plt.xlabel('x')

  plt.legend()
  plt.grid()

  plt.savefig('sectioning.png', dpi=150)
  plt.show()


# --
# Main function
if __name__ == '__main__':

  # objective function
  fx = lambda x: np.sin(x) - np.sin(10/3 * x)

  # x space for function
  x = np.linspace(0, 5, 400)

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
  xs, sec_list = sectioning(a, b)
  print("Sectioning minimum: ", xs)

  # plot
  plot_sectioning(x, sec_list)
  



