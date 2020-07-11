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


def plot_onsets(x, fs, N, hop, c, thresh, onsets_times):
  """
  onset plots
  """

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # frame vector
  time_frames = (np.arange(0, len(x) - (N / hop - 1) * hop, hop) + N / 2) / fs 

  # plot
  plt.figure(3, figsize=(8, 4))
  plt.plot(t, x / max(x), label='audiofile', linewidth=1)
  plt.plot(time_frames, c / max(c), label='complex', linewidth=1)
  plt.plot(time_frames, thresh / max(c), label='adapt thresh', linewidth=1)

  # draw onsets
  for i, a in enumerate(onset_times):
    plt.axvline(x=float(a), dashes=(5, 1), color='k')


  plt.ylabel('magnitude')
  plt.xlabel('time [s]')
  plt.grid()
  plt.legend()

  plt.show()


if __name__ == '__main__':
  """
  main
  """

  # audio file names
  #files = ['./ignore/ghs/ab_r4-6.mp3']
  files = ['./ignore/sounds/eight7.wav', './ignore/sounds/bed0.wav']

  # run through all files
  for file in files:

    print("---sound: ", file)

    # load file
    x, fs = librosa.load(file, mono=True)

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

    # calc chroma
    chroma = calc_chroma(x, fs, hop, n_octaves=4, bins_per_octave=36, fmin=librosa.note_to_hz('C3'))


    # --
    # onsets

    # calc onsets
    onsets, onset_times, c, thresh = calc_onsets(x, fs, N=N, hop=hop, adapt_frames=5, adapt_alpha=0.1, adapt_beta=1)


    # --
    # plots

    #plot_whole_chroma(chroma, fs, hop, fmin, bins_per_octave=12)
    plot_onsets(x, fs, N, hop, c, thresh, onset_times)




















