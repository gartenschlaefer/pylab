# --
# chorus detection

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix

from mpl_toolkits.mplot3d import Axes3D

# my personal mia lib
from mia import *

from scipy.io import loadmat

def plot_pca(x_pca, c):
  """
  plot pca in 2d and 3d
  """
  # colors mixing
  c1 = 'green'
  c2 = 'orange'

  # 2d
  plt.figure(1)

  plt.scatter(x_pca[c=='K', 0], x_pca[c=='K', 1], color=color_fader(c1, c2, 0), marker='o', label='Kick')
  plt.scatter(x_pca[c=='S', 0], x_pca[c=='S', 1], color=color_fader(c1, c2, 0.5), marker='v', label='Snare')
  plt.scatter(x_pca[c=='H', 0], x_pca[c=='H', 1], color=color_fader(c1, c2, 1), marker='s', label='Hi-Hat')

  plt.grid()
  plt.legend()
  plt.tight_layout()

  # 3d
  fig = plt.figure(2)
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(x_pca[c=='K', 0], x_pca[c=='K', 1], x_pca[c=='K', 2], color=color_fader(c1, c2, 0), marker='o', label='Kick')
  ax.scatter(x_pca[c=='S', 0], x_pca[c=='S', 1], x_pca[c=='S', 2], color=color_fader(c1, c2, 0.5), marker='v', label='Snare')
  ax.scatter(x_pca[c=='H', 0], x_pca[c=='H', 1], x_pca[c=='H', 2], color=color_fader(c1, c2, 1), marker='s', label='Hi-Hat')

  plt.legend()
  plt.tight_layout()

  plt.show()


# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'

  # lda mat files
  file_names = ['DrumFeatures.mat']


  # run through all files
  for file_name in file_names:

    print("---sound: ", file_name)

    # load file
    drum_feat = loadmat(file_dir + file_name)

    x = drum_feat['drumFeatures'][0][0][0]
    y = drum_feat['drumFeatures'][0][0][1]

    print(x.shape)
    print(y.shape)

    # --
    # visualize features with pca

    x_pca = calc_pca(x)

    # compare with sklearn
    #pca = PCA(n_components=3)
    #x_pca2 = pca.fit_transform(x, y_true)

    print("x_pca: ", x_pca.shape)
    #plot_pca(x_pca, c)


    # --
    # lda

    # sklearn approach
    clf = LDA()
    clf.fit(x, y)

    y_hat = clf.predict(x)

    print("LDA predict: ", clf.predict(x))

    cm = confusion_matrix(y, y_hat)

    print("confusion matrix:\n", cm)



    






















