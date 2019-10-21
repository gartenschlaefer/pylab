import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

from mia import *



# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'
  file_name = 'a_l_c1.wav'
  fs, x = wavfile.read(file_dir + file_name)

  print("--fs: ", fs)

  # sample params
  N = 1024

  # windowing params
  ol = N // 2
  hop = N - ol

  n_frames = len(x) / hop
  print("n_frames: ", n_frames)

  # some vectors
  f = np.arange(0, fs/2, fs/N)

  # windowing
  w = np.hanning(N)

  # apply windows
  x_buff = np.multiply(w, buffer(x, N, ol))

  # transformation matrix
  H = np.exp(1j * 2 * np.pi / N * np.outer(np.arange(N), np.arange(N)))

  # transfored signal
  X = np.dot(x_buff, H)

  # log
  Y = 20 * np.log10(2 / N * np.abs(X[:, 0:512]))

  # some frame to plot
  fi = int(n_frames // 3) 

  # peaks layout with three partial tones
  peaks = np.array([0, 0, 0, 0])
  #f_est = np.array([0, 0, 0, 0])

  # save all peaks
  for fi in np.arange(Y.shape[0]):

    # find peaks
    p, _ = signal.find_peaks(Y[fi], height=(40, 100))

    # check if all three partial tones are included 
    if p.size != 0 and p[0:4].shape[0] == 4:

      # save peaks
      peaks = np.vstack((peaks, p[0:4]))

      # save instantaneous frequency estimation
      #f_est = np.vstack((f_est, inst_f(X, fi, p[0:4], hop, N, fs)))

  # save frequency estimation to file
  # np.save('f_est', f_est)
  
  # load inst frequency
  f_est = np.load('f_est.npy')

  # remove first entry
  peaks = np.delete(peaks, 0, 0)
  f_est = np.delete(f_est, 0, 0)


  print("-- frequency:")
  print("mean peaks f: ", np.mean(peaks, 0) * fs / N)
  print("mean estimated f: ", np.mean(f_est, 0))

  # frequency should be at c1 = 261,626Hz

  # plot something
  # plt.figure(12)
  # #plt.subplot(211)
  # plt.plot(f, Y[fi], label='bariton')
  # plt.scatter(p[0:4] * fs / N, Y[fi][p[0:4]], label='peaks')

  # plt.grid()
  # plt.xlabel('frequency')
  # plt.ylabel('magnitude')
  # plt.legend()
  # plt.show()


  # plot spectogramm
  # f, t, Sxx = signal.spectrogram(x, fs, nperseg=2048, mode='magnitude')
  # plt.pcolormesh(t, f, Sxx)
  # plt.ylabel('Frequency [Hz]')
  # plt.xlabel('Time [sec]')
  # plt.ylim(0, 3000)
  # plt.show()

  plt.figure(1)
  plt.specgram(x, 1024, fs)
  plt.ylim(0, 2000)
  plt.show()
