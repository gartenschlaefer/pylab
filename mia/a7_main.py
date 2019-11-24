import numpy as np
import matplotlib.pyplot as plt
import librosa

from mia import *
from get_annotations import get_annotations
from erb_filter import erb_filter_bank

from scipy import signal
from scipy.io import wavfile

# midi
from pretty_midi import note_number_to_name
from pretty_midi import note_number_to_hz

from get_onset_mat import get_onset_mat



def plot_filter(b, a, fs):

  # check filters
  w, h = signal.freqz(b, a)

  plt.figure(10)
  plt.plot(w / np.pi * fs/2, 20 * np.log10(abs(h)))
  plt.ylabel('magnitude [dB]')
  plt.xlabel('frequency [Hz]')

  plt.xscale('log')
  plt.ylim([-60, 10])

  plt.grid()
  plt.show()


  plt.figure(10)
  plt.plot(w / np.pi * fs/2, np.angle(h))
  plt.ylabel('Angle [rad]')
  plt.xlabel('frequency [Hz]')

  plt.xscale('log')

  plt.grid()
  plt.show()



# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/mpitch/'

  # audio file names
  file_names = ['01-AchGottundHerr_4Kanal.wav']

  # file name to mat file
  mat_file_name = '01-AchGottundHerr-GTF0s.mat'

  # var name
  var_name = 'GTF0s'



  # window length
  N = 512

  # windowing params
  ol = N // 2
  hop = N - ol

  # run through all files
  for file_name in file_names:

    print("---sound: ", file_name)

    # load file
    #fs, x = wavfile.read(file_dir + file_name)
    x, fs = librosa.load(file_dir + file_name, sr=22050, mono=False)

    # print some signal stuff
    print("x: ", x.shape)
    print("fs: ", fs)
    print("frame length: ", len(x) // hop)

    # gets onsets
    onsets, m = get_onset_mat(file_dir + mat_file_name, var_name)
    print("onsets: ", onsets.shape)

    
    # --
    # snythesis of channels


    # --
    # pre whitening

    # warped linear predictive coding

    # print warped lambda
    print(warped_lambda(fs))

    # allpass
    b, a = allpass_coeffs(warped_lambda(fs))
    #plot_filter(b, a, fs)


    # --
    # channel filtering

    # high frequency channel
    # high pass params
    order = 4
    f_cut = 1000

    # high pass filter
    b, a = signal.butter(order, f_cut / (fs/2), 'high')
    x_h = signal.filtfilt(b, a, x)

    # half wave rectification
    x_h = x_h.clip(min=0)

    # low pass filtering
    b, a = signal.butter(order, f_cut / (fs/2), 'low')
    x_h = signal.filtfilt(b, a, x_h)

    # low frequency channel
    # band pass params
    order = 4
    f_cut = np.array([70, 1000])

    # low pass filter
    b, a = signal.butter(order, f_cut / (fs/2), btype='band')
    x_l = signal.filtfilt(b, a, x)
    #plot_filter(b, a, fs)

    print("xl: ", x_l.shape)


    # --
    # Autocorrelation in the frequency domain


    # --
    # SACF (summed autocorrelation function)


    # --
    # ESACF (enhance autocorrelation function)














