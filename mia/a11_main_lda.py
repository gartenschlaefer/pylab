# --
# chorus detection

import numpy as np
import matplotlib.pyplot as plt

# my personal mia lib
from mia import *

from scipy.io import loadmat


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
    c = drum_feat['drumFeatures'][0][0][1]

    print(x.shape)
    print(c.shape)

    # visualize features with pca


















