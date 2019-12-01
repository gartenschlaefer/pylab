import numpy as np
import matplotlib.pyplot as plt
import librosa

from mia import *

from scipy import signal
from scipy.io import wavfile



# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'

  # audio file names
  file_names = ['megalovania.wav']

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

    X_white = adaptive_whitening(X)

    f = np.linspace(0, fs/2, N//2)

    plt.figure(1, figsize=(8, 4))
    plt.plot(f, np.abs(X[2000, 0:N//2]), label='origin')
    plt.plot(f, np.abs(X_white[2000, :]), label='whitened')

    plt.ylabel('magnitude')
    plt.xlabel('frequency [Hz]')

    plt.grid()
    plt.legend()

    #plt.savefig('adaptive_whitening' + '.png', dpi=150)
    plt.show()

    # adaptive whitening



    
















