# --
# Tempo Estimation

import numpy as np
import matplotlib.pyplot as plt
import librosa

from mia import *

from scipy import signal
from scipy.io import wavfile


def plot_mel_weights(w_f, w_mel, fs, N):

  mel_f = np.linspace(0, f_to_mel(fs / 2), N)
  f = np.linspace(0, fs / 2, N)

  # mel weights
  plt.figure(4, figsize=(8, 4))
  for w in w_mel:
    plt.plot(mel_f, w, label='mel weights')
    #plt.plot(w, label='mel weights')
  plt.ylabel('magnitude')
  plt.xlabel('mel')
  #plt.savefig('weights_mel.png', dpi=150)
  plt.show()

  # f weights
  plt.figure(4, figsize=(8, 4))
  for w in w_f:
    plt.plot(f, w, label='f weights')
    #plt.plot(w, label='f weights')
  plt.ylabel('magnitude')
  plt.xlabel('frequency [Hz]')
  #plt.savefig('weights_f.png', dpi=150)
  plt.show()


def plot_odf(fs, N, hop, x, sf):
  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)
  t_frame = (np.arange(0, len(x) - (N / hop - 1) * hop, hop) + N / 2) / fs 

  plt.figure(1, figsize=(8, 4))
  plt.plot(t, x/max(x), label='audio')
  plt.plot(t_frame, sf/max(sf), label='spectral flux')

  plt.ylabel('magnitude')
  plt.xlabel('time [s]')

  plt.grid()
  plt.legend()

  #plt.savefig('adaptive_whitening' + '.png', dpi=150)
  plt.show()


def plot_filt_log(X_filt, X_log):

  plt.figure(1, figsize=(8, 4))
  plt.plot(X_filt[10, :], label='filtered')
  plt.plot(X_log[10, :], label='log')

  plt.ylabel('magnitude')
  plt.xlabel('frequency band')

  plt.grid()
  plt.legend()

  #plt.savefig('adaptive_whitening' + '.png', dpi=150)
  plt.show()


def plot_adaptive_whitening(fs, N, X, X_white):

  f = np.linspace(0, fs/2, N//2)

  plt.figure(1, figsize=(8, 4))
  plt.plot(f, np.abs(X[10, 0:N//2]), label='origin')
  plt.plot(f, np.abs(X_white[10, :]), label='whitened')

  plt.ylabel('magnitude')
  plt.xlabel('frequency [Hz]')

  plt.grid()
  plt.legend()

  #plt.savefig('adaptive_whitening' + '.png', dpi=150)
  plt.show()


# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'

  # audio file names
  #file_names = ['megalovania.wav', 'kick300.wav', 'pluck400.wav']
  file_names = ['kick300.wav', 'pluck400.wav']

  # window length
  N = 1024

  # run through all files
  for file_name in file_names:

    print("---sound: ", file_name)

    # load file
    x, fs = librosa.load(file_dir + file_name, sr=22050, mono=True)

    # windowing params
    t_hop = 0.01
    hop = int(t_hop * fs)
    ol = N - hop

    # print some signal stuff
    print("x: ", x.shape)
    print("fs: ", fs)
    print("hop: ", hop)
    print("frame length: ", len(x) // hop)

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


    # --
    # Preprocessing

    # adaptive whitening
    X_white = adaptive_whitening(X)

    # mel filterbank
    # filter bands
    M_bands = 20
    mel_bands = np.linspace(0, f_to_mel(fs / 2), M_bands)
    #print("mel bands: ", mel_bands)

    # weights
    w_f, w_mel, n_bands = mel_band_weights(M_bands, fs, N//2)

    # sum the weighted energies
    X_filt = np.inner(np.abs(X_white), w_f)

    # log magnitude
    X_log = np.log(2 * X_filt + 1)


    # --
    # Onset detection function

    # spectral flux
    sf = spectral_flux(X_log)

    # sampling rate for odf
    fso = t_hop

    # frames
    n_frames = len(sf)


    # --
    # Tempogram

    # create tatums
    tatums = np.linspace(0.06, 0.43, 50)
    print("selected tatums: ", tatums)

    # weighting with complex function
    W = np.exp(-1j * 2 * np.pi / fso * np.outer(1 / tatums, np.arange(n_frames)))

    # weighted odf
    odf_w = np.transpose(sf * W)
    print("odf_w: ", odf_w.shape)

    # length of odf frames
    L = 150

    # windowing
    w = np.hanning(L)

    odf_buff = buffer2D(odf_w, L)

    print("odf buff:", odf_buff.shape)

    # Tempogram measure
    M = np.dot(w, buffer2D(odf_w, L))
    print("M: ", M.shape)

    # modified tempogram

    # phase
    phi = np.angle(M)

    # phase diff
    d_phi = np.diff(phi, axis=0)

    # modification
    Mp = M[:-1] * (1 - np.abs(d_phi))**100

    # normalization
    Mp = Mp / np.max(Mp)

    #print("phi", phi.shape)
    #print("d_phi", d_phi.shape)
    #print("Mp", Mp.shape)





    # --
    # plots

    #plot_mel_weights(w_f, w_mel, fs, N)

    #plot_adaptive_whitening(fs, N, X, X_white)

    #plot_filt_log(X_filt, X_log)

    #plot_odf(fs, N, hop, x, sf)


    plt.figure(6)
    plt.title(file_name)
    plt.imshow(np.transpose(np.abs(M)), aspect='auto', extent=[0, len(x)/fs, tatums[0]*1000, tatums[-1]*1000, ])
    #plt.imshow(np.transpose(np.abs(Mp)), aspect='auto', extent=[0, len(x)/fs, tatums[0]*1000, tatums[-1]*1000, ])
    plt.ylabel('tatum [ms]')
    plt.xlabel('time [s]')

    plt.colorbar()
    plt.savefig('tempo_' + file_name.split('.')[0] + '.png', dpi=150)
    plt.show()
















    
















