import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal
#from scipy.io import wavfile
import librosa

from mia import *


def plot_wavefile(x, fs, name):

  # some vectors
  t = np.arange(0, len(x)/fs, 1/fs)

  plt.figure(1, figsize=(8, 4))
  plt.plot(t, x, label='audiofile')

  plt.title(name)
  plt.ylabel('magnitude')
  plt.xlabel('time [s]')

  plt.grid()
  #plt.legend()
  plt.savefig(name + '.png', dpi=150)
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
    plot_wavefile(x, fs, file_name)
    