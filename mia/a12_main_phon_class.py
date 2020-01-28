# --
# phoneme classification

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# some lean tools
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

# knn
from sklearn import neighbors

# my personal mia lib
from mia import *
from get_phonemlabels import get_phonemlabels


def gmm(x, y):
  """
  gaussian mixture model
  """

  # make classifier
  clf = GaussianMixture(n_components=3, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=200, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)

  # fit
  clf.fit(x, y)

  # predict
  y_hat = clf.predict(x)

  # label encoder
  le = preprocessing.LabelEncoder()
  le.fit(y)

  return le.inverse_transform(y_hat)


def knn(x, y, n_neighbors=15):
  """
  knn classifier
  """

  # step size in the mesh
  h = .1  

  # Create color maps
  cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
  cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

  #weight_types = ['uniform', 'distance']
  weight_types = ['uniform']

  # for weights in weight_types:

  #   print("weights: ", weights)

  #   # we create an instance of Neighbours Classifier and fit the data.
  #   clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

  #   print("clf done: ", weights)
  #   clf.fit(x, y)
  #   print("fit done: ", weights)

  #   # Plot the decision boundary
  #   x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
  #   y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

  #   xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  #   # predict
  #   Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  #   print("predict done: ", weights)

  #   # Put the result into a color plot
  #   Z = Z.reshape(xx.shape)
  #   plt.figure()
  #   plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

  #   # Plot also the training points
  #   plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

  #   plt.xlim(xx.min(), xx.max())
  #   plt.ylim(yy.min(), yy.max())
  #   plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors, weights))

  # plt.show()

  # classifier
  clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight_types[0])

  # train
  clf.fit(x, y)

  # predict
  y_hat = clf.predict(x)

  return y_hat


def plot_features(zcr, rms, qff, labels):
  """
  plot all features against each other
  """

  # rms, zcr
  plt.figure()
  plt.scatter(zcr[labels=='P'], rms[labels=='P'], label='P')
  plt.scatter(zcr[labels=='C'], rms[labels=='C'], label='C')
  plt.scatter(zcr[labels=='V'], rms[labels=='V'], label='V')

  plt.ylabel('RMS value')
  plt.xlabel('Zero Crossing Rate')
  plt.legend()

  plt.grid()
  plt.show()

  # rms, qff
  plt.figure()
  plt.scatter(qff[labels=='P'], rms[labels=='P'], label='P')
  plt.scatter(qff[labels=='C'], rms[labels=='C'], label='C')
  plt.scatter(qff[labels=='V'], rms[labels=='V'], label='V')

  plt.ylabel('RMS value')
  plt.xlabel('First Formant')
  plt.legend()

  plt.grid()
  plt.show()

  # zcr, qff
  plt.figure()
  plt.scatter(qff[labels=='P'], zcr[labels=='P'], label='P')
  plt.scatter(qff[labels=='C'], zcr[labels=='C'], label='C')
  plt.scatter(qff[labels=='V'], zcr[labels=='V'], label='V')

  plt.ylabel('Zero Crossing')
  plt.xlabel('First Formant')
  plt.legend()

  plt.grid()
  plt.show()


# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'

  # lda mat files
  file_names = ['A0101B.wav']

  # annotation file
  anno_file = "A0101_Phonems.txt"

  # run through all files
  for file_name in file_names:

    # load file
    x, fs = librosa.load(file_dir + file_name, sr=22050, mono=True)

    # sample params
    N = 1024
    ol = N // 2
    hop = N - ol

    # get phonems
    phonems = get_phonemlabels(file_dir + anno_file)

    # just print them
    #for i, phone in enumerate(phonems):
    #  print("phone: ", i, phone)

    # extract code
    consonant_code_list = np.array(['-', 'D', 'N', 'Q', 'S', 'T', 'Z', 'b', 'd', 'dZ', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'tS', 'v', 'w', 'z'])
    vowel_code_list = np.array(['I', 'E', '{', 'A', 'V', 'U', 'i', 'e', 'e@', 'u', 'o', 'O:', 'aI', 'OI', 'aU', '3`', '@', '@`', 'u:', 'eI', '3:', 'i:', 'A:', '@U'])
    pause_code_list = np.array(['_'])
    end_code = np.array(['-'])

    # get phon_samples
    phon_samples = (np.array([phonems[:, 0], phonems[:, 1]]).astype(float) * fs).astype(int)
    print("phon_samples: ", phon_samples.shape)

    # get labels
    labels = phonems[:, 2]

    # replace it by label codes
    labels[np.isin(labels, pause_code_list)] = 'P'
    labels[np.isin(labels, consonant_code_list)] = 'C'
    labels[np.isin(labels, vowel_code_list)] = 'V'

    #print(labels)

    # init stuff
    zcr = np.zeros(len(labels))
    rms = np.zeros(len(labels))
    qff = np.zeros(len(labels))

    # run through each phonem save its features
    for i, label in enumerate(labels):

      # get
      x_i = x[phon_samples[0, i] : phon_samples[1, i]]

      # print sample info
      #print("shape: ", x_i.shape)
      #print("label: ", label)

      # window
      w = np.hanning(len(x_i))

      # calculate zcr feature
      zcr[i] = zero_crossing_rate(x_i, w, axis=0)

      # calculate root mean
      rms[i] = calc_rms(x_i, w)

      # first formant
      #qff[i] = q_first_formant(x_i, w, fs, f_roi=[300, 1000])

    # plot features
    #plot_features(zcr, rms, qff, labels)


    # --
    # train and test set

    print("Amount of features in dataset P: {}, C: {}, V: {}".format(np.sum(labels=='P'), np.sum(labels=='C'), np.sum(labels=='V')))


    # feature set [samples x features]
    x_data = np.array([zcr, rms]).T
    print("x_data: ", x_data.shape)

    # trainings data
    #x_train = x_data[0:100, :]
    #y_train = labels[0:100]

    x_train = x_data
    y_train = labels
    print("Amount of features in train P: {}, C: {}, V: {}".format(np.sum(y_train=='P'), np.sum(y_train=='C'), np.sum(y_train=='V')))


    # --
    # classification
    
    # KNN
    y_train_knn_pred = knn(x_train, y_train, n_neighbors=10)

    # Gaussian Mixture Model
    y_train_gmm_pred = gmm(x_train, y_train)

    #print("y_train_pred gmm: ", y_train_pred)
    acc_knn, cp, fp = calc_accuracy(y_train_knn_pred, y_train)
    acc_gmm, cp, fp = calc_accuracy(y_train_gmm_pred, y_train)

    print("\n-----Accuracy\n KNN: [{}], GMM: [{}]".format(acc_knn, acc_gmm))








    






















