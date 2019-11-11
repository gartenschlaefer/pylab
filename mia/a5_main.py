import numpy as np
import matplotlib.pyplot as plt
import librosa

from mia import *
from get_annotations import get_annotations

# --
def plot_wavefile(x, fs, name, annotation_file, save=False):


  #anno_text = ['part 1', 'part 2', 'part 3']

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # plot
  plt.figure(1, figsize=(8, 4))
  plt.plot(t, x, label='audiofile', linewidth=1)



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

  #file_names = ['megalovania.wav']
  file_names = ['imperial_march_plastic_trumpet.wav']

  #annotation_file_names = ['megalovania.txt', 'megalovania_parts.txt']
  annotation_file_names = ['imperial_march_plastic_trumpet.txt']

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

    print("x: ", x.shape)
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

    # adaptive threshold
    thresh = adaptive_threshold(c, H=5)

    # get onsets from measure and threshold
    onset = thresholding_onset(c, thresh)

    # annotations
    anno = get_annotations(file_dir + annotation_file_names[0])

    # calculate onset times
    onset_times = (onset * np.arange(0, len(onset)) * hop + hop/2) / fs 
    onset_times = onset_times[onset_times > hop / 2 / fs]

    # score measure
    score = score_onset_detection(onset_times, anno, tolerance=0.02)


    # --
    # awesome plot

    # time vector
    t = np.arange(0, len(x)/fs, 1/fs)

    # frame vector
    time_frames = (np.arange(0, len(x) - hop, hop) + hop / 2) / fs 

    # plot
    plt.figure(3)
    plt.plot(t, x / max(x), label='audiofile', linewidth=1)
    plt.plot(time_frames, c / max(c), label='complex domain', linewidth=1)
    plt.plot(time_frames, thresh / max(c), label='adaptive threshold', linewidth=1)

    #plt.plot(time_frames, onset, label='onset', linewidth=1)

    # annotations labels
    for i, a in enumerate(anno):
      # draw vertical lines
      plt.axvline(x=float(a), dashes=(2, 2), color='k')


    # annotations targets
    for i, a in enumerate(onset_times):
      # draw vertical lines
      plt.axvline(x=float(a), dashes=(5, 1), color='g')

    plt.title('complex domain onset')
    plt.ylabel('magnitude')
    plt.xlabel('time [s]')

    plt.legend()
    plt.show()















