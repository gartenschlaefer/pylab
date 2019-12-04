import numpy as np
import matplotlib.pyplot as plt
from np_table import np_table


def color_fader(c1, c2, mix):
  """
  fades colors
  """
  import matplotlib as mpl

  # convert colors to rgb
  c1 = np.array(mpl.colors.to_rgb(c1))
  c2 = np.array(mpl.colors.to_rgb(c2))

  return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def get_brackets(fx, g, x0, s, k = 2):
  """
  get brackets of minimizing function fx(x)
  """

  # first step
  a = x0
  b = x0 + g * s

  a_list = [a]
  b_list = [b]

  # check if downhill direction
  if fx(b) > fx(a):

    # wrong direction
    print('no minima in this direction -> change dir')
    g = -g

  # init
  stop_cond = False
  f_calls = 0

  # bracket algorithm
  while not stop_cond:

    # step
    c = b + g * s

    # add a function call
    f_calls += 1

    # brackets found
    if fx(c) > fx(b):

      # update b
      b = c

      # update lists
      a_list.append(a)
      b_list.append(b)

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

      # update lists
      a_list.append(a)
      b_list.append(b)

      # update step size
      s = s * k;

  return (a, b, f_calls, a_list, b_list)


def sectioning(a, b, sec_lim=1e-3):
  """
  sectioning of the brackets with the golden ration
  """
  gold = (1 + np.sqrt(5)) / 2

  # list of sections for print
  sec_list = np.empty((0, len(a)), float)

  # function calls
  f_calls = 0

  while np.linalg.norm(b - a) > sec_lim:

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

    # update function calls
    f_calls += 1


  return (a + b) / 2, sec_list, f_calls


def plot_function(name, fx, x):
  
  # plot
  plt.figure(1, figsize=(8, 4))

  # plot obj function
  plt.plot(x, fx(x))

  plt.ylabel('f(x)')
  plt.xlabel('x')

  plt.grid()

  plt.savefig(name + '.png', dpi=150)
  plt.show()



def plot_simple(fx, sec_list=[], a_list=[], b_list=[]):
  """
  plot a simple 2D function
  """

  x = np.linspace(-5, 5, 400)

  # plot
  plt.figure(1, figsize=(8, 4))

  # plot obj function
  plt.plot(x, fx(x))

  # colors mixing
  c1 = 'green'
  c2 = 'orange'
  c_mix = np.linspace(0, 1, len(a_list))
  a_mix = np.linspace(0.5, 1, len(a_list))

  # plot brackets
  for i, sec in enumerate(sec_list):
    c = color_fader(c1, c2, c_mix[i])
    plt.axvline(x=sec[0], color=c, dashes=(4, 2), alpha=a_mix[i], label='sec'+str(i+1))
    plt.axvline(x=sec[1], color=c, dashes=(4, 2), alpha=a_mix[i])

  # plot steps
  for i, a in enumerate(a_list):
    c = color_fader(c1, c2, c_mix[i])
    plt.scatter(a, fx(a), color=c, marker='*')

  if b_list:
    plt.scatter(b_list[-1], fx(b_list[-1]), color='orange', marker='*')

  # limit for brackets
  #plt.ylim((-0.8, -0.2))
  #plt.xlim((a_list[0]-0.1, b_list[-1]+0.1))

  plt.ylabel('f(x)')
  plt.xlabel('x')

  #plt.legend()
  plt.grid()

  plt.savefig('sectioning.png', dpi=150)
  plt.show()


def plot_rosenbrock(fx_rosen, a_list=[], b_list=[]):
  """
  plot the famous rosenbrock
  """

  # colors mixing
  c1 = 'green'
  c2 = 'orange'
  c_mix = np.linspace(0, 1, len(a_list))
  a_mix = np.linspace(0.5, 1, len(a_list))

  # mesh of data
  x = np.meshgrid(np.linspace(-3.5, 3.5, 100), np.linspace(-3.5, 3.5, 100))
  x1 = x[0]
  x2 = x[1]

  # discrete values for rosenbrock
  fx = fx_rosen(x)

  # plot rosenbrock
  plt.figure(20)

  plt.contour(x1, x2, fx, np.logspace(-0.5,  3.5, 20, base=10), cmap = 'gray')
  plt.contour(x1, x2, fx, 50, cmap = 'jet')
  #plt.contour(x1, x2, fx, 50, cmap = 'gray')
  #plt.contourf(x1, x2, fx, 100, cmap = 'jet')

  # plot steps
  for i, a in enumerate(a_list):
    c = color_fader(c1, c2, c_mix[i])
    plt.scatter(a[0], a[1], color=c, marker='*')

  if b_list:
    plt.scatter(b_list[-1][0], b_list[-1][1], color='orange', marker='*')

  plt.xlabel('x1')
  plt.ylabel('x2')

  plt.savefig('rosen' + '.png', dpi=150)

  plt.show()


class ModelTester():
  """
  class for testing different things
  """

  def __init__(self, fx, name):

    # objective function
    self.fx = fx
    self.name = name
    print("---ModelTester_init: " + name + "---")


  def run_line_search_bs(self, g, x0, step_sizes, k_factors):
    """
    runs the line search algorithm with
    brackets and sectioning
    """

    print("--Line search with brackets and sectioning--")

    # check input - integer
    if isinstance(step_sizes, (int, float)):
      print("sorry list is requirde")
      return

    # check input - equal len
    if len(step_sizes) != len(k_factors):
      print("parameter length are not equal")
      return

    # print header
    print("params:\t | s \t| k \t| f(a) \t\t| f(b) \t\t| f(b)-f(a) \t| a \t\t| b \t\t| b-a \t\t| f-calls \t| x_min \t|")
    
    # list for printing
    table_list = np.empty((0, 10))

    # run over param list
    for s, k in zip(step_sizes, k_factors):

      # get the brackets
      a, b, f_calls_b, a_list, b_list = get_brackets(self.fx, g, x0, s, k)

      # sectioning
      xs, sec_list, f_calls_s = sectioning(a, b)

      # update list
      table_list = np.vstack((table_list, (s, k, self.fx(a), self.fx(b), self.fx(b) - self.fx(a), str(a), str(b), str(b-a), f_calls_b, str(xs))))

      # for printing of numpy arrays -> precision
      np.set_printoptions(precision=4)

      # print stuff
      print("results: | {:.2f}\t| {:d}\t".format(s, k) + 
        "| {:.4f}\t| {:.4f}\t".format(float(self.fx(a)), float(self.fx(b))) + 
        "| {:.4f}\t| {}\t| {} \t".format(float(self.fx(a) - self.fx(b)), a, b) +
        "| {}\t| {:d}\t\t| {}\t|".format(b - a, f_calls_b, xs))

    header = ['step size s', 'mult. factor k', 'f(a)', 'f(b)', 'f(b)-f(a)', 'a', 'b', 'b-a', 'function calls', 'x min']
    
    # make table
    np_table(self.name, table_list, header=header)



# --
# Main function
if __name__ == '__main__':

  # --
  # objective functions

  # objective function with sine waves
  fx_sine = lambda x: np.sin(x) - np.sin(10/3 * x)

  # rosenbrock function
  fx_rosen = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


  # --
  # parameters for testing

  # step size
  s = np.array([0.01, 0.1, 1, 0.01, 0.1, 1])

  # step size multiplication factor
  k = np.array([2, 2, 2, 4, 4, 4])


  # --
  # line search univariate -> sine wave task1

  # starting point
  x0 = 0.0

  # search direction
  g = np.array([1])

  # Test the line search model with univariate function
  model_tester = ModelTester(fx_sine, 'univariate')
  model_tester.run_line_search_bs(g, x0, s, k)


  # --
  # rosenbrock task2

  # starting point
  x0 = np.array([3, -2]) 

  # search direction
  g = np.array([-1, 1])

  # Test the line search model with the rosenbrock
  model_tester = ModelTester(fx_rosen, 'rosenbrock')
  model_tester.run_line_search_bs(g, x0, s, k)


  # --
  # plot functions

  #a, b, f_calls_b, a_list, b_list = get_brackets(fx_sine, g, x0, 0.01, 2)
  # sectioning
  #xs, sec_list, f_calls_s = sectioning(a, b)
  #print("x_star: ", xs)
  # plot simple function
  #plot_simple(fx_sine, sec_list=[], a_list=a_list, b_list=b_list)

  #a, b, f_calls_b, a_list, b_list = get_brackets(fx_rosen, g, x0, 0.01, 2)
  # sectioning
  #xs, sec_list, f_calls_s = sectioning(a, b)
  # plot rosenbrock
  #plot_rosenbrock(fx_rosen, a_list, b_list)
  plot_rosenbrock(fx_rosen)

  # plot sine function
  #plot_function('obj_fun_sine', fx_sine, np.linspace(-5, 5, 400),)






