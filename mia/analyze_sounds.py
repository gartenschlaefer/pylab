# --
# chorus detection

import numpy as np
import matplotlib.pyplot as plt
import re

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


def plot_onsets(x, fs, N, hop, c, thresh, onsets_times, plot_path, name='None'):
  """
  onset plots
  """

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # frame vector
  time_frames = (np.arange(0, len(x) - (N / hop - 1) * hop, hop) + N / 2) / fs 
  #time_frames = np.arange(0, len(x) * hop + N / 2) / fs 
  #(onsets * np.arange(0, len(onsets)) * hop + N / 2) / fs
  time_frames = time_frames[:len(c)]

  # plot
  plt.figure(figsize=(9, 5))
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

  plt.savefig(plot_path + name + '.png', dpi=150)



def extract_label(file):
  """
  extract file label
  """

  # extract filename
  #file_name = re.findall(r'[\w+ 0-9]+\.wav', file)[0]
  file_name = re.findall(r'[\w+ 0-9]+\.mp3', file)[0]

  # extract label from filename
  label = re.sub(r'([0-9]+\.wav)', '', file_name)

  # extract file index from filename
  #file_index = re.sub(r'[a-z A-Z]|(\.wav)', '', file_name)

  print("label: ", label)

  return label


if __name__ == '__main__':
  """
  main
  """

  # audio file names
  #files = ['./ignore/ghs/ab_r4-6.mp3']
  #files = ['./ignore/sounds/eight7.wav', './ignore/sounds/bed0.wav']
  #files = ['./ignore/sounds/left3.wav', './ignore/sounds/right6.wav']
  #files = ['./ignore/ghs/HB_05.mp3']
  #files = ['./ignore/ghs/HB_06.mp3']
  #files = ['./ignore/ghs/HB_07.mp3']
  #files = ['./ignore/ghs/HB_08.mp3']
  #files = ['./ignore/ghs/HB_09.mp3']
  files = ['./ignore/ghs/HB_14.mp3']

  plot_path = './ignore/plots/'

  # run through all files
  for file in files:

    print("---sound: ", file)

    # sampling rate
    fs = 22050

    # mfcc window and hop size
    #N, hop = int(0.025 * fs), int(0.010 * fs)
    N, hop = 1024, 512

    # load file
    x, fs = librosa.load(file, mono=True, sr=fs)

    # print some signal stuff
    print("x: ", x.shape)
    print("fs: ", fs)
    print("hop: ", hop)
    print("frame length: ", len(x) // hop)

    # extract filename label
    label = extract_label(file)


    # --
    # chroma features

    # calc chroma
    fmin = librosa.note_to_hz('C2')
    chroma = calc_chroma(x, fs, hop, n_octaves=4, bins_per_octave=36, fmin=fmin)


    # --
    # onsets

    h, a, b = 5, 0.09, 0.7

    param_str = '{}_h-{}_a-{}_b-{}'.format(label, h, a, b).replace('.', 'p')

    # calc onsets
    #onsets, onset_times, c, thresh = calc_onsets(x, fs, N=N, hop=hop, adapt_frames=h, adapt_alpha=a, adapt_beta=b)


    # --
    # plots

    plot_whole_chroma(chroma, fs, hop, fmin, bins_per_octave=12)
    #plot_onsets(x, fs, N, hop, c, thresh, onset_times, plot_path, name= 'onsets_' + param_str)


  # show
  plt.show()
















