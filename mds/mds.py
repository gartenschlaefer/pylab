"""
mds tests
sources:
  https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
"""

from sklearn import datasets, manifold
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
from matplotlib import ticker
import mpl_toolkits.mplot3d
import numpy as np

# unused but required import for doing 3d projections with matplotlib < 3.2



def mds_s_curve():
  """
  s-curve
  """

  # samples
  n_samples = 1500
  S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)

  # 3d plot
  #plot_3d(S_points, S_color, "Original S-curve samples")

  # scaling defines
  n_neighbors = 12
  n_components = 2

  assert n_components <= 2

  # mds scaling
  md_scaling = manifold.MDS(n_components=n_components, max_iter=50, n_init=4, random_state=0, normalized_stress=False)
  S_scaling = md_scaling.fit_transform(S_points)

  # determine plot function
  f_plot = plot_2d if n_components == 2 else plot_1d

  # plot
  f_plot(S_scaling, S_color, "Multidimensional scaling")

  #plot_2d(S_scaling, S_color, "Multidimensional scaling")
  #plot_1d(S_scaling, S_color, "Multidimensional scaling")


def plot_3d(points, points_color, title):
  """
  3d plot
  """
  x, y, z = points.T

  fig, ax = plt.subplots(
      figsize=(6, 6),
      facecolor="white",
      tight_layout=True,
      subplot_kw={"projection": "3d"},
  )
  fig.suptitle(title, size=16)
  col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
  ax.view_init(azim=-60, elev=9)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

  fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
  plt.show()


def plot_2d(points, points_color, title):
  fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
  fig.suptitle(title, size=16)
  add_2d_scatter(ax, points, points_color)
  plt.show()


def plot_1d(points, points_color, title):
  fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
  fig.suptitle(title, size=16)
  add_1d_scatter(ax, points, points_color)
  plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
  x, y = points.T
  ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
  ax.set_title(title)
  ax.xaxis.set_major_formatter(ticker.NullFormatter())
  ax.yaxis.set_major_formatter(ticker.NullFormatter())


def add_1d_scatter(ax, points, points_color, title=None):
  x = points.T
  y = np.zeros(x.shape)
  print(x.shape)
  print(y.shape)
  ax.scatter(x, np.zeros(x.shape), c=points_color, s=50, alpha=0.8)
  ax.set_title(title)
  ax.xaxis.set_major_formatter(ticker.NullFormatter())
  ax.yaxis.set_major_formatter(ticker.NullFormatter())


def mds_digits():
  """
  digits
  """

  # load data
  x, _ = datasets.load_digits(return_X_y=True)
  print("x: ", x.shape)

  # create embeddings
  embedding = MDS(n_components=2, normalized_stress='auto')
  x_t = embedding.fit_transform(x[:100])
  print("x_t: ", x_t.shape)


if __name__ == '__main__':
  """
  main
  """

  # s-curve
  mds_s_curve()



