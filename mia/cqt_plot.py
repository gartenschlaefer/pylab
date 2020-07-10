# --
# chorus detection

import numpy as np
import matplotlib.pyplot as plt

# librosa
import librosa
import librosa.display

# filter stuff
from scipy import signal

# my personal mia lib
from mia import *


def plot_whole_chroma(C, fs, hop, fmin, bins_per_octave, annotation_file=[], x_axis='time'):
  """
  plot whole song
  """
  plt.figure(2, figsize=(8, 4))
  librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=fs, hop_length=hop, x_axis=x_axis, y_axis='chroma', fmin=fmin, bins_per_octave=bins_per_octave)
  plt.colorbar(format='%+2.0f dB')
  #plt.title('Constant-Q power spectrum')

  # add anno
  if annotation_file:
    plot_add_anno(annotation_file, text_height=10)

  plt.tight_layout()
  plt.show()


# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/ghs/'

  # audio file names
  file_names = ['ab_r4-6.mp3']

  # run through all files
  for file_name in file_names:

    print("---sound: ", file_name)

    # load file
    x, fs = librosa.load(file_dir + file_name, mono=True)

    # windowing params
    N = 1024
    hop = 512
    ol = N - hop

    # print some signal stuff
    print("x: ", x.shape)
    print("fs: ", fs)
    print("hop: ", hop)
    print("frame length: ", len(x) // hop)


    # --
    # chroma features

    n_octaves = 4
    bins_per_octave = 36
    fmin = librosa.note_to_hz('C3')

    chroma = calc_chroma(x, fs, hop, n_octaves, bins_per_octave, fmin)
    plot_whole_chroma(chroma, fs, hop, fmin, bins_per_octave=12)




















