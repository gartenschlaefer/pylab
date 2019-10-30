import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal
#from scipy.io import wavfile
import librosa

from mia import *


def plot_wavefile(x, fs):

  # some vectors
  t = np.arange(0, len(x)/fs, 1/fs)

  plt.figure(1)
  plt.plot(t, x)
  plt.show()


# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'
  file_names = ('bass-drum-kick.wav', 'cymbal.wav', 'hihat-closed.wav', 'snare.wav')

  # run through all files
  for file_name in file_names:

    print("---sound: ", file_name)

    # load file
    x, fs = librosa.load(file_dir + file_name)

    print("fs: ", fs)

    # plot wave file
    plot_wavefile(x, fs)
    