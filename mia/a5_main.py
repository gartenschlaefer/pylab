import numpy as np
import matplotlib.pyplot as plt
import librosa

from mia import *
from get_annotations import get_annotations

# --
def plot_wavefile(x, fs, name, annotation_file, save=False):

  # parts
  annotations = get_annotations(annotation_file)
  #anno_text = ['part 1', 'part 2', 'part 3']

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # plot
  plt.figure(1, figsize=(8, 4))
  plt.plot(t, x, label='audiofile', linewidth=1)

  # annotations
  for i, a in enumerate(annotations):

    # draw vertical lines
    plt.axvline(x=float(a), dashes=(1, 1), color='k')

    # write labels
    #plt.text(x=float(a), y=0.85, s=anno_text[i], color='k', fontweight='semibold')

  plt.title(name)
  plt.ylabel('magnitude')
  plt.xlabel('time [s]')
  plt.xlim((0, 1))

  plt.grid()
  #plt.legend()
  if save:
    plt.savefig(name.split('.')[0] + '.png', dpi=150)
    
  plt.show()


def primitive_strategies(x, x_buff, N, fs, hop, X):

  # windowing
  w = np.hanning(N)

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # frame vector
  time_frames = (np.arange(0, len(x) - hop, hop) + hop / 2) / fs 

  # short time energy
  ste = st_energy(x_buff, w)
  
  # amplitude difference in stfts
  a_diff = amplitude_diff(X, N)

  # plot
  plt.figure(2)
  plt.plot(t, x / max(x), label='audiofile', linewidth=1)
  plt.plot(time_frames[:-1], a_diff / max(a_diff), label='ampl diff', linewidth=1)
  plt.plot(time_frames, ste / max(ste), label='ste', linewidth=1)

  plt.title('primitive strategies')
  plt.ylabel('magnitude')
  plt.xlabel('time [s]')

  plt.legend()
  plt.show()


# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'
  file_names = ['megalovania.wav']

  annotation_file_names = ['megalovania.txt', 'megalovania_parts.txt']

  # window length
  N = 512

  # windowing params
  ol = N // 2
  hop = N - ol

  # run through all files
  for file_name in file_names:

    print("---sound: ", file_name)

    # load file
    x, fs = librosa.load(file_dir + file_name)

    print("fs: ", fs)
    n_frames = len(x) // hop
    print("frame length: ", n_frames)

    # plot wavefile
    #plot_wavefile(x, fs, file_name, file_dir + annotation_file_names[0], save=True)

    # --
    # STFT

    # windowing
    w = np.hanning(N)

    # apply windows
    x_buff = np.multiply(w, buffer(x, N, ol))

    # transformation matrix
    H = np.exp(1j * 2 * np.pi / N * np.outer(np.arange(N), np.arange(N)))

    # transformed signal
    X = np.dot(x_buff, H)

    # log
    #Y = 20 * np.log10(2 / N * np.abs(X[:, 0:N//2]))

    # some primitive strategies
    #primitive_strategies(x, x_buff, N, fs, hop, X)

    # phase deviation
    #phase_deviation(X, N)

    # complex domain
    c = complex_domain_onset(X, N)

    thres = adaptive_threshold(c)


    # time vector
    t = np.arange(0, len(x)/fs, 1/fs)


    # frame vector
    time_frames = (np.arange(0, len(x) - hop, hop) + hop / 2) / fs 

    plt.figure(3)
    plt.plot(t, x / max(x), label='audiofile', linewidth=1)
    plt.plot(time_frames, c / max(c), label='complex domain', linewidth=1)
    plt.plot(time_frames, thres / max(c), label='complex domain', linewidth=1)

    plt.title('complex domain onset')
    plt.ylabel('magnitude')
    plt.xlabel('time [s]')

    plt.legend()
    plt.show()















