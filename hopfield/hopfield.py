# --
# fourier transform lab

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import yaml
import cv2

from pathlib import Path
from glob import glob

# lambda function
softmax = lambda x : np.exp(x) / np.sum(np.exp(x))


def plot_image(img):
  """
  plot image
  """

  # setup figure
  fig = plt.figure()

  # create axis
  ax = plt.axes()

  # plot
  im = ax.imshow(img)
  plt.show()


def image_pre_processing(cfg, img):
  """
  image pre processing
  """

  # ravel image and sign it
  img = img.ravel()

  # finished if continuous
  if cfg['is_continuous']: 

    # make float and [0, 1]
    img = img.astype(float)
    img /= np.max(img)
    return img

  # bi-polar states
  img = np.array([1 if x else -1 for x in r_img]).astype(int)

  return r_img


def get_data(cfg):
  """
  get data
  """

  # file extension
  file_ext = ".pgm" if cfg['is_continuous'] else ".pbm"

  # train files
  train_files = sorted(glob(cfg['dataset_path'] + cfg['train_folder'] + '*' + file_ext))
  retrieve_files = sorted(glob(cfg['dataset_path'] + cfg['retrieve_folder'] + '*' + file_ext))

  print(train_files)
  print(retrieve_files)

  # trainings images
  train_imgs = np.empty((0, 16 * 16))
  retrieve_imgs = np.empty((0, 16 * 16))

  # load training data
  for train_file in train_files:

    # read file
    img = cv2.imread(train_file, cv2.IMREAD_GRAYSCALE)

    # pre-processing
    img = image_pre_processing(cfg, img)

    # stack
    train_imgs = np.vstack((train_imgs, img[np.newaxis, :]))

  retrieve_file_names = []
  for retrieve_file in retrieve_files:

    # add name
    name = Path(retrieve_file).name
    retrieve_file_names.append(name)

    # read file
    img = cv2.imread(retrieve_file, cv2.IMREAD_GRAYSCALE)

    # pre-processing
    img = image_pre_processing(cfg, img)

    # stack
    retrieve_imgs = np.vstack((retrieve_imgs, img[np.newaxis, :]))

  return train_imgs, retrieve_imgs, retrieve_file_names


class Hopfield():
  """
  hopfield
  """

  def __init__(self, net_type='modern', data_type='continuous', beta=1.0):
    """
    init
    """

    # weights and bias
    self.W = None
    self.b = None
    self.xi = None
    self.beta = beta

    # stored patterns
    self.X = None

    # bi-polar
    if data_type == 'bi-polar':

      # function setting
      self.f_train = self.bi_polar_modern_train
      self.f_predict_update = self.bi_polar_modern_predict_update

    # continuous
    else:

      # function setting
      self.f_train = self.continuous_modern_train
      self.f_predict_update = self.continuous_modern_predict_update

  def train(self, train_samples):
    """
    training function
    """
    self.f_train(train_samples)


  def predict_update(self, xi):
    """
    training function
    """
    return self.f_predict_update(xi)


  def classic_train(self, train_samples):
    """
    training
    """

    # dim
    N, dim = train_samples.shape

    # define W
    self.W = np.zeros((dim, dim))

    for x in train_samples:
      self.W += np.outer(x, x)


  def classic_predict_update(self, x):
    """
    classic update
    """
    return np.sign(self.W @ x)


  def bi_polar_modern_train(self, train_samples):
    """
    modern training, keep training samples
    """
    N, dim = train_samples.shape
    # assume quadratic shape
    d = int(np.sqrt(dim))

    self.X = train_samples
    self.W = self.X.reshape(N * d, d)


  def bi_polar_modern_predict_update(self, x):
    """
    modern update
    """

    # prediction
    y = np.zeros(x.shape)

    for i, x_i in enumerate(x):

      x_pos = x.copy()
      x_neg = x.copy()
      x_pos[i] = 1
      x_neg[i] = -1

      y[i] = np.sign( np.sum(np.exp(self.X @ x_pos)) - np.sum(np.exp(self.X @ x_neg)) ) 

    return y


  def continuous_modern_train(self, train_samples):
    """
    modern training, keep training samples
    """
    # same as bi-polar
    self.bi_polar_modern_train(train_samples)


  def continuous_modern_predict_update(self, x):
    """
    modern continuous update
    """
    y = self.X.T @ softmax(self.beta * self.X @ x)

    return y




if __name__ == '__main__':
  """
  main
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  print(cfg)

  # get data
  train_imgs, retrieve_imgs, retrieve_file_names = get_data(cfg)

  # create net
  hopfield = Hopfield(beta=0.5)

  # train
  hopfield.train(train_imgs)

  # plot weight matrix
  plot_image(hopfield.W)

  # retrieve images
  for retrieve_img, name in zip(retrieve_imgs, retrieve_file_names):

    print("file to predict: ", name)
    y = retrieve_img
    y_prev = y
    plot_image(y.reshape(16, 16))

    # epochs
    for epoch in range(20):

      # predict
      y = hopfield.predict_update(y)

      # stop condition
      if (np.isclose(y, y_prev, rtol=1e-02, atol=1e-03,)).all(): 
        print("converged!")
        break

      # copy
      y_prev = y.copy()

      # plot prediction
      plot_image(y.reshape(16, 16))