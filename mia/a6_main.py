import numpy as np
import matplotlib.pyplot as plt
import librosa

from mia import *
from get_annotations import get_annotations
from erb_filter import erb_filter_bank


def plot_onsets(x, fs, hop, c, thresh, onset_times, file_name):
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



# delete later
def plot_filter_bank(x, fs):

  N = int(x.shape[1])

  # frequency vector
  f = np.arange(0, fs/2, fs/N)

  # plot
  plt.figure(1, figsize=(8, 4))
  Y = 20 * np.log10(np.abs(np.fft.fft(x)))[:, 0:N//2]
  plt.plot(f, np.transpose(Y))

  plt.xscale('log')
  plt.ylabel('magnitude [dB]')
  plt.xlabel('frequency [Hz]')

  plt.grid()
  #plt.savefig('erb_filter_bank.png', dpi=150)
  plt.show()



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
    onsets = thresholding_onset(c, thresh)

    # calculate onset times
    onset_times = (onsets * np.arange(0, len(onsets)) * hop + N / 2) / fs 
    onset_times = onset_times[onset_times > N / 2 / fs]
    


    # plot onsets
    #plot_onsets(x, fs, hop, c, thresh, onset_times, file_name)


    # --
    # get time interval of onsets

    # onset deltas to onset samples num
    onset_samples = np.where(onsets == 1)[0] * hop + N / 2
    onset_samples = onset_samples.astype(int)

    # time interval
    t_i = 0.05

    # samples for interval
    N_i = int(t_i * fs)

    print("N_i:", N_i)

    # init onset matrix
    x_of = np.zeros((len(onset_samples), N_i))

    # windowing
    w = np.hanning(N_i)

    # window onset frames
    for i, on in enumerate(onset_samples):
      
      # onset frame
      x_of[i, :] = x[on:on+N_i] * w

      # input response
      #x_of[i, :] = np.zeros(N_i)
      #x_of[i, 0] = 1

    # --
    # ERBs filter bank

    # params
    n_bands = 10
    f_low = 100
    f_high = 5000

    # filter bank
    y, fc = erb_filter_bank(x_of, fs, n_bands, f_low, f_high)

    # y is of shape [ch, onset, num_samples]
    print("size of y: ", y.shape)

    #plot_filter_bank(y[:, 2, :], fs)
    # simple hair cell model

    # Autocorrelogram




