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


def plot_fisher_ration(fisher_ratios, compare_labels):

  c1 = 'green'
  c2 = 'orange'
  fade = [0, 0.5, 1]
  
  plt.figure(2, figsize=(8, 4))
  
  for i, r in enumerate(fisher_ratios.T):
    plt.plot(r, label=compare_labels[i], color=color_fader(c1, c2, fade[i]))

  plt.xlabel('features')
  plt.ylabel('fisher ratio')

  plt.grid()
  plt.minorticks_on()
  plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
  
  plt.legend()
  plt.tight_layout()

  plt.show()


def plot_lda_transformed_2D(x_h, y):
  """
  plot lda transformed in 1D and 2D
  """
  # colors mixing
  c1 = 'green'
  c2 = 'orange'

  # 1D
  plt.figure(3)

  plt.scatter(x_h[y=='K', 0], np.zeros(len(x_h[y=='K', 0])), color=color_fader(c1, c2, 0), marker='o', label='Kick')
  plt.scatter(x_h[y=='S', 0], np.zeros(len(x_h[y=='S', 0])), color=color_fader(c1, c2, 0.5), marker='v', label='Snare')
  plt.scatter(x_h[y=='H', 0], np.zeros(len(x_h[y=='H', 0])), color=color_fader(c1, c2, 1), marker='s', label='Hi-Hat')

  plt.grid()
  plt.legend()
  plt.tight_layout()
  
  # 2d
  plt.figure(4)

  plt.scatter(x_h[y=='K', 0], x_h[y=='K', 1], color=color_fader(c1, c2, 0), marker='o', label='Kick')
  plt.scatter(x_h[y=='S', 0], x_h[y=='S', 1], color=color_fader(c1, c2, 0.5), marker='v', label='Snare')
  plt.scatter(x_h[y=='H', 0], x_h[y=='H', 1], color=color_fader(c1, c2, 1), marker='s', label='Hi-Hat')

  plt.grid()
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

    # shuffle a bit
    r = np.random.permutation(len(y))

    x = x[r]
    y = y[r]

    #print(y)

    # --
    # visualize features with pca

    x_pca = calc_pca(x)

    # compare with sklearn
    #pca = PCA(n_components=3)
    #x_pca2 = pca.fit_transform(x, y_true)

    print("x_pca: ", x_pca.shape)
    #plot_pca(x_pca, y)


    # --
    # lda sklearn approach

    # init
    clf = LDA()
    clf.fit(x, y)

    y_hat = clf.predict(x)

    cm = confusion_matrix(y, y_hat)
    print("confusion matrix:\n", cm)


    # --
    # compute fisher ratio for feature analysis

    fisher_ratios, compare_labels = calc_fisher_ration(x, y)
    #plot_fisher_ration(fisher_ratios, compare_labels)


    # --
    # my lda

    n_lda_dim = 1
    w, bias, x_hat, label_list = lda_classifier(x, y, method='class_dependent', n_lda_dim=n_lda_dim)
    
    # print lda stuff
    print("w", w.shape)
    print("bias", bias.shape)
    print("x_hat: ", x_hat.shape)
    print("label_list", label_list)

    # init predictions
    x_h = np.zeros((len(label_list), n_lda_dim))
    y_h = []

    # run through each sample
    for xn in x:

      for lda_dim in range(n_lda_dim):

        # classify sample
        x_h[:, lda_dim] = w[:, :, lda_dim] @ xn.T - bias

      # calculate distance of lda dimensions
      d = np.linalg.norm(x_h, axis=1)

      # add minimum from distance to prediction
      y_h.append(label_list[np.argmin(d)])

    # compute confusion matrix
    cm = confusion_matrix(y, y_h)
    print("confusion matrix:\n", cm)








    






















