import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal
#from scipy.io import wavfile
import librosa

from mia import *


def plot_wavefile(x, fs, name, save):

  # some vectors
  t = np.arange(0, len(x)/fs, 1/fs)

  plt.figure(1, figsize=(8, 4))
  plt.plot(t, x, label='audiofile')

  plt.title(name)
  plt.ylabel('magnitude')
  plt.xlabel('time [s]')

  plt.grid()
  #plt.legend()
  if save:
    plt.savefig(name + '.png', dpi=150)
    
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


def plot_weights(w_f, w_mel, fs, N):

  mel_f = np.linspace(0, f_to_mel(fs / 2), N)
  f = np.linspace(0, fs / 2, N)

  # mel weights
  plt.figure(4, figsize=(8, 4))
  for w in w_mel:
    plt.plot(mel_f, w, label='mel weights')
  plt.ylabel('magnitude')
  plt.xlabel('mel')
  #plt.savefig('weights_mel.png', dpi=150)
  plt.show()

  # f weights
  plt.figure(4, figsize=(8, 4))
  for w in w_f:
    plt.plot(f, w, label='f weights')
  plt.ylabel('magnitude')
  plt.xlabel('frequency [Hz]')
  #plt.savefig('weights_f.png', dpi=150)
  plt.show()


def plot_mfcc_frame(mfcc, name):

  plt.figure(5)
  for m in range(4):
    plt.plot(mfcc[m, :], label='mfcc' + str(m))
  plt.title(name)
  plt.ylabel('magnitude')
  plt.xlabel('filter band')
  plt.legend(title='from frames:')
  plt.ylim((-50, 50))
  plt.grid()
  plt.savefig('mfcc_frames' + name + '.png', dpi=150)
  plt.show()


def plot_mfcc_spec(mfcc, name):
  plt.figure(6)
  plt.title(file_name)
  plt.imshow(np.transpose(mfcc), aspect='auto', vmin=-20, vmax=40)
  plt.ylabel('filter band')
  plt.xlabel('time frame')
  plt.colorbar()
  plt.savefig('mfcc_spec' + name + '.png', dpi=150)
  plt.show()

# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'
  file_names = ['bass-drum-kick.wav', 'cymbal.wav', 'hihat-closed.wav', 'snare.wav']

  # window length
  N = 512

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
    Y = 20 * np.log10(2 / N * np.abs(X[:, 0:N//2]))


    # --
    # MFCC

    # filter bands
    M = 8
    mel_bands = np.linspace(0, f_to_mel(fs / 2), M)
    print("mel bands: ", mel_bands)

    # weights
    w_f, w_mel, n_bands = mel_band_weights(M, fs, N//2)

    # plot the weights
    #plot_weights(w_f, w_mel, fs, N//2)

    # energy of fft
    E = np.power(2 / N * np.abs(X[:, 0:N//2]), 2)

    # sum the weighted energies
    u = np.inner(E, w_f[0:-1, :])

    # log
    u = np.log(u)
    #print(u)

    # discrete cosine transform
    mfcc = dct(u, n_bands - 1)

    print("mfcc: ", mfcc.shape)
    #print("mfcc: ", mfcc)

    plot_mfcc_frame(mfcc, file_name)
    plot_mfcc_spec(mfcc, file_name)


    
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

    