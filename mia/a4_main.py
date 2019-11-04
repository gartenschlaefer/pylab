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
  #plt.savefig(name + '.png', dpi=150)
  plt.show()


def plot_frequency_response(x, fs, name):

  # some vectors
  f = np.arange(0, fs/2, fs/N)

  plt.figure(2, figsize=(8, 4))
  plt.plot(f, x, label='frequency response')

  plt.title(name)
  plt.ylabel('magnitude [dB]')
  plt.xlabel('frequency [Hz]')

  plt.grid()
  #plt.legend()
  #plt.savefig(name + '.png', dpi=150)
  plt.show()


def plot_cepstrum(x, fs, name):

  # some vectors
  q = np.arange(0, len(x)/fs, 1/fs)

  plt.figure(2, figsize=(8, 4))
  plt.plot(q, x, label='cepstrum')

  plt.title(name)
  plt.ylabel('magnitude')
  plt.xlabel('quefrency [s]')

  plt.grid()
  #plt.legend()
  #plt.savefig(name + '.png', dpi=150)
  plt.show()


def plot_mel_transform():

  f = np.linspace(0, 10000, 10000)
  mel = f_to_mel(f)

  plt.figure(3, figsize=(8, 4))
  plt.plot(f, mel, label='mel')

  plt.ylabel('mel')
  plt.xlabel('frequency [Hz]')

  plt.grid()
  #plt.legend()
  #plt.savefig('mel_transform.png', dpi=150)
  plt.show()



# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'
  #file_names = ('bass-drum-kick.wav', 'cymbal.wav', 'hihat-closed.wav', 'snare.wav')
  file_names = ('bass-drum-kick.wav', 'cymbal.wav')

  # window length
  N = 1024

  # windowing params
  ol = N // 2
  hop = N - ol

  # check if mel transform works
  #plot_mel_transform()

  # run through all files
  for file_name in file_names:

    print("---sound: ", file_name)

    # load file
    x, fs = librosa.load(file_dir + file_name)

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
    Y = 20 * np.log10(2 / N * np.abs(X[:, 0:512]))

    # --
    # MFCC



    # filter bands
    M = 8

    #print("mel bands: ", m)
    mel_band_weights(M, fs, N)

    Ex = np.power(2 / N * np.abs(X[:, 0:512]), 2)


    # -- 
    # cepstrum

    #plot_cepstrum(cepstrum(x_buff[0], N)[0:512], N, file_name)



    # --
    # plot stuff

    # plot wave file
    #plot_wavefile(x, fs, file_name)

    # plot frequency response
    #plot_frequency_response(Y[n_frames // 2], fs, file_name)

    # sample params

    