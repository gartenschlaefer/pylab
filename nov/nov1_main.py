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


def get_brackets(fx, x0, s, k = 2):
  """
  get brackets of minimizing function fx(x)
  """

  # search direction gradient
  g = 1

  # first step
  a = x0
  b = x0 + g * s

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

  return (a, b, f_calls)


def sectioning(a, b, sec_lim=1e-3):
  """
  sectioning of the brackets with the golden ration
  """
  gold = (1 + np.sqrt(5)) / 2

  # list of sections for print
  sec_list = np.empty((0, 2), float)

  # function calls
  f_calls = 0

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


class ModelTester():
  """
  class for testing different things
  """

  def __init__(self, fx):

    # objective function
    self.fx = fx
    print("---ModelTester_init---")


  def run_line_search_bs(self, x0, step_sizes, k_factors):
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

      #print(s)
      #print(k)

      
      # get the brackets
      a, b, f_calls_b = get_brackets(self.fx, x0, s, k)

      # sectioning
      xs, sec_list, f_calls_s = sectioning(a, b)

      # update list
      table_list = np.vstack((table_list, (s, k, self.fx(a), self.fx(b), self.fx(b) - self.fx(a), a, b, b-a, f_calls_b, xs)))

      # print stuff
      print("results: | {:.2f}\t| {:d}\t".format(s, k) + 
        "| {:.4f}\t| {:.4f}\t".format(self.fx(a), self.fx(b)) + 
        "| {:.4f}\t| {:.4f}\t| {:.4f} \t".format(self.fx(a) - self.fx(b), a, b) +
        "| {:.4f}\t| {:d}\t\t| {:.4f}\t|".format(b - a, f_calls_b, xs))

    header = ['step size s', 'mult. factor k', 'f(a)', 'f(b)', 'f(b)-f(a)', 'a', 'b', 'b-a', 'function calls', 'x min']
    
    # make table
    #np_table('univar_obj_fun', table_list, header=header)



# --
# Main function
if __name__ == '__main__':

  # objective function with sine waves
  fx_sine = lambda x: np.sin(x) - np.sin(10/3 * x)

  # rosenbrock function
  fx_rosen = lambda x1, x2: 100 * (x2 - x1**2)**2 + (1 - x1)**2

  # plot the function
  #plot_function('obj_fun_sine', fx_sine, np.linspace(-5, 5, 400))


  # plot

  # calc rosenbrock
  x1, x2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

  fx = fx_rosen(x1, x2)

  print(fx.shape)

  # plot rosenbrock
  plt.figure(20)

  #X, Y = meshgrid(linspace(-.5, 2., 100), linspace(-1.5, 4., 100))
  #Z = rosenbrockfunction(X, Y)
  plt.contour(x1, x2, fx, np.logspace(-0.5,  3.5, 20, base = 10), cmap = 'gray')
  #plt.contour(x1, x2, fx, 50, cmap = 'jet')
  #plt.contour(x1, x2, fx, 50, cmap = 'gray')
  plt.contourf(x1, x2, fx, 100, cmap = 'jet')

  plt.title('Rosenbrock Function: $f(x,y) = (1-x)^2+100(y-x^2)^2$')
  plt.xlabel('x')
  plt.ylabel('y')


  #plt.plot(x, fx(x))
  #plt.ylabel('f(x)')
  #plt.xlabel('x')
  #plt.grid()
  #plt.savefig(name + '.png', dpi=150)

  plt.show()

  # --
  # line search

  # starting point
  x0 = 0.0

  # step size
  s = np.array([0.01, 0.1, 1, 0.01, 0.1, 1])

  # step size multiplication factor
  k = np.array([2, 2, 2, 4, 4, 4])

  # get the brackets
  #a, b, f_calls_b = get_brackets(fx_sine, x0, s, k)

  # sectioning
  #xs, sec_list, f_calls_s = sectioning(a, b)
  #print("Sectioning minimum: ", xs)

  # plot
  #plot_sectioning(x, sec_list)

  # Test the line Search model
  model_tester = ModelTester(fx_sine)

  model_tester.run_line_search_bs(x0, s, k)
  



