# --
# single pitch detection

import numpy as np
import matplotlib.pyplot as plt
import librosa

from mia import *
from get_annotations import get_annotations
from erb_filter import erb_filter_bank
from scipy import signal

# midi
from pretty_midi import note_number_to_name
from pretty_midi import note_number_to_hz



def plot_erbs_onset(file_name, onset_samples, N_i, fs, x_of, y, fc, oni=0):

  # time vector
  t = np.arange(onset_samples[oni], onset_samples[oni] + N_i) / fs

  plt.figure(1, figsize=(8, 5))

  # offset to signal
  of = max(x_of[oni, :])

  for ch, c in enumerate(fc):

    # hop vertically for center frequencies
    hop_up = (len(fc) - ch) / 100

    # plot fills
    plt.fill_between(t, hop_up + of, y[ch, oni, :] + hop_up + of, label='erbs fc={:.0f}Hz'.format(c))

  # print signal
  plt.plot(t, x_of[oni, :], label='x')

  plt.title(file_name + ' at the {}. predicted onset'.format(oni+1))
  plt.ylabel('magnitude')
  plt.xlabel('time [s]')

  plt.grid()
  plt.legend(loc='upper left', prop={'size': 7})
  #plt.savefig(file_name.split('.')[0] + '_erbs_onset.png', dpi=150)
  plt.show()



def plot_autocorrelogram(file_name, onset_samples, N_i, fs, x_of, y, fc, oni=0, peaks=[]):

  # time vector
  t = np.arange(onset_samples[oni], onset_samples[oni] + N_i) / fs

  plt.figure(1, figsize=(8, 5))

  # offset to signal
  of = max(x_of[oni, :])

  for ch, c in enumerate(fc):
    # hop vertically for center frequencies
    hop_up = (len(fc) - ch) / 100
    #plt.plot(t, y[ch, oni, :] + ch/100, label='erbs')
    plt.fill_between(t, hop_up + of, y[ch, oni, :] + hop_up + of, label='corr erbs fc={:.0f}Hz'.format(c))

  plt.plot(t, x_of[oni, :], label='sum')

  # print peaks
  if not len(peaks) == 0:
    for p in peaks[oni, :]:
      plt.axvline(x=p/fs + t[0], dashes=(2, 2), color='k')

  plt.title(file_name + ' at the {}. predicted onset'.format(oni+1))
  plt.ylabel('magnitude')
  plt.xlabel('lag [s]')

  plt.grid()
  plt.legend(prop={'size': 7})
  #plt.savefig(file_name.split('.')[0] + '_autocorrelogram.png', dpi=150)
  plt.show()



def plot_onsets(x, fs, hop, c, thresh, onset_times, file_name, midi_onsets, f_est, tolerance=0.02):
    # --
    # awesome plot

    # time vector
    t = np.arange(0, len(x)/fs, 1/fs)

    # frame vector
    time_frames = (np.arange(0, len(x) - hop, hop) + hop / 2) / fs 

    # plot
    plt.figure(3, figsize=(8, 4))
    plt.plot(t, x / max(x), label='audiofile', linewidth=1)
    plt.plot(time_frames, c / max(c), label='complex domain', linewidth=1)
    plt.plot(time_frames, thresh / max(c), label='adaptive threshold', linewidth=1)

    # annotations midi labels
    for i, mid in enumerate(midi_onsets):

      # draw vertical lines
      if i == 0: 
        # put label to legend
        plt.axvline(x=mid[3], dashes=(2, 2), color='k', label="midi-labels")
        #plt.text(x=mid[3], y=0.9, s=note_number_to_name(mid[1]), color='k', fontweight='semibold')
        plt.text(x=mid[3], y=0.9, s=round(note_number_to_hz(mid[1]), 1), color='k', fontweight='semibold')
        
      else:
        plt.axvline(x=float(mid[3]), dashes=(2, 2), color='k')
        #plt.text(x=mid[3], y=0.9, s=note_number_to_name(mid[1]), color='k', fontweight='semibold')
        plt.text(x=mid[3], y=0.9, s=round(note_number_to_hz(mid[1]), 1), color='k', fontweight='semibold')

    # tolerance band of each label
    neg_label_tolerance = midi_onsets[:, 3] - tolerance
    pos_label_tolerance = midi_onsets[:, 3] + tolerance
    green_label = False
    red_label = False

    # annotations targets
    for i, a in enumerate(onset_times):

      # decide if correct or not -> color
      is_tp = np.sum(np.logical_and(neg_label_tolerance < a, pos_label_tolerance > a))

      # draw vertical lines
      if is_tp == 1:

        if green_label == False: 
          # put label
          green_label = True
          plt.axvline(x=float(a), dashes=(5, 1), color='g', label="targets TP")
          plt.text(x=float(a), y=1, s=f_est[i], color='g', fontweight='semibold')
        else:
          plt.axvline(x=float(a), dashes=(5, 1), color='g')
          plt.text(x=float(a), y=1, s=f_est[i], color='g', fontweight='semibold')

      else:
        if red_label == False: 
          # put label
          red_label = True
          plt.axvline(x=float(a), dashes=(5, 1), color='r', label="targets FP")
          plt.text(x=float(a), y=0.8, s=f_est[i], color='r', fontweight='semibold')
        else:
          plt.axvline(x=float(a), dashes=(5, 1), color='r')
          plt.text(x=float(a), y=0.8, s=f_est[i], color='r', fontweight='semibold')

    plt.title(file_name)
    plt.ylabel('magnitude')
    plt.xlabel('time [s]')

    plt.grid()
    plt.legend(prop={'size': 7})

    #plt.savefig('class' + str(r) + '.png', dpi=150)
    plt.show()



# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/f0/'

  file_names = ['BWV847.wav', 'guitar_riff_1.wav', 'guitar_riff_2.wav']
  #file_names = ['guitar_riff_1.wav']


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

    # print some signal stuff
    print("x: ", x.shape)
    print("fs: ", fs)
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

      # boundary
      if (on + N_i + N // 2) < len(x):

        # onset frame
        x_of[i, :] = x[on+N//2:on+N_i+N//2] * w


    # --
    # ERBs filter bank

    # params
    n_bands = 12
    f_low = 80
    f_high = 4000

    # filter bank
    y, fc = erb_filter_bank(x_of, fs, n_bands, f_low, f_high)


    # --
    # simple hair cell model

    # rectification
    y_hc = y.clip(min=0)

    # low pass filtering

    # low pass filter params
    order = 4
    f_cut = 1000

    # low pass filter
    b, a = signal.butter(order, f_cut / (fs/2), 'low')
    y_hc = signal.filtfilt(b, a, y_hc)


    # --
    # Autocorrelogram

    y_cor = np.zeros(y_hc.shape)

    # all channels
    for ch in np.arange(len(fc)):

      # all onsets
      for oni in np.arange(len(onset_samples)):

        # correlation
        y_cor[ch, oni, :] = np.correlate(y_hc[ch, oni, :], y_hc[ch, oni, :], mode='full')[N_i-1:]


    # --
    # Summary of the Autocorrelogram

    # sum over all channels
    y_sum = np.sum(y_cor, axis=0)


    # --
    # Pitch detection

    # estimated frequency
    f_est = np.zeros(len(onset_samples))
    peaks = np.zeros((len(onset_samples), 5))

    # find peaks:
    for oni in np.arange(len(onset_samples)):

      h_max = max(y_sum[oni, :])

      # find peaks
      p, v = signal.find_peaks(y_sum[oni, :], height=(0.3 * h_max, h_max))

      # get second highes peak
      if not len(v['peak_heights']) == 0:
        v_max_idx = np.argsort(v['peak_heights'])[-1]

        # difference from highest to second highest peak
        diff_samples = p[v_max_idx]

        # frequency estimate
        f_est[oni] = round(fs / diff_samples, 1)

      else: 
        f_est[oni] = 0


    # print onset index
    #oni = 2

    # plot stuff
    #plot_erbs_onset(file_name, onset_samples, N_i, fs, x_of, y_hc, fc, oni=oni)
    #plot_autocorrelogram(file_name, onset_samples, N_i, fs, y_sum, y_cor, fc, oni=oni)


    # --
    # Midi comparison

    # get midi events
    midi_events = get_midi_events(file_dir + file_name.split('.')[0] + '.mid')

    # only onsets
    midi_onsets = midi_events[midi_events[:, 0] == 1]

    # plot onsets
    plot_onsets(x, fs, hop, c, thresh, onset_times, file_name, midi_onsets, f_est, tolerance=0.02)




