import numpy as np
import matplotlib.pyplot as plt
import librosa

from mia import *
from get_annotations import get_annotations


# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/f0/'

  #file_names = ['BWV846M_P_sel.wav', 'BWV847.wav', 'guitar_riff_1.wav', 'guitar_riff_2.wav']
  file_names = ['guitar_riff_1.wav']


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

    # --
    # onset detection

    # complex domain
    c = complex_domain_onset(X, N)

    # adaptive threshold
    thresh = adaptive_threshold(c, H=5)

    # get onsets from measure and threshold
    onset = thresholding_onset(c, thresh)


    
    # calculate onset times
    onset_times = (onset * np.arange(0, len(onset)) * hop + hop/2) / fs 
    onset_times = onset_times[onset_times > hop / 2 / fs]


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

    # annotations targets
    for i, a in enumerate(onset_times):
      # draw vertical lines
      plt.axvline(x=float(a), dashes=(5, 1), color='g')

    plt.title(file_name)
    plt.ylabel('magnitude')
    plt.xlabel('time [s]')

    plt.legend()
    plt.show()