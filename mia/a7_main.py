# --
# multi pitch detection

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

# get onsets from bach
from get_onset_mat import get_onset_mat

# lpc stuff
from audiolazy.lazy_lpc import levinson_durbin 


def plot_filter(b, a, fs):

  # check filters
  w, h = signal.freqz(b, a)

  # frequency response
  plt.figure(10)
  plt.plot(w / np.pi * fs/2, 20 * np.log10(abs(h)))
  plt.ylabel('magnitude [dB]')
  plt.xlabel('frequency [Hz]')

  plt.xscale('log')
  plt.ylim([-60, 10])

  plt.grid()
  plt.show()

  # phase response
  plt.figure(10)
  plt.plot(w / np.pi * fs/2, np.angle(h))
  plt.ylabel('Angle [rad]')
  plt.xlabel('frequency [Hz]')

  plt.xscale('log')

  plt.grid()
  plt.show()


def plot_wlpc(x_of, y_hat_rosa, y_hat_warp, fs):

  # plot stuff
  plt.figure(1, figsize=(8, 4))
  #plt.plot(lpc_rosa)
  plt.plot(x_of, label='x at onset')
  plt.plot(y_hat_rosa, label='lpc estimate')
  plt.plot(y_hat_warp, label='wlpc estimate')
  plt.ylabel('magnitude')
  plt.xlabel('samples')
  plt.legend()
  plt.grid()
  plt.savefig('wlpc_time.png', dpi=150)
  plt.show()

  plt.figure(1, figsize=(8, 4))
  #plt.plot(lpc_rosa)
  
  N = len(x_of)
  f = np.linspace(0, fs / 2, N // 2)

  plt.plot(f, 20 * np.log10(np.abs(np.fft.fft(x_of)))[0:N//2], label='X at onset')
  plt.plot(f, 20 * np.log10(np.abs(np.fft.fft(y_hat_rosa)))[0:N//2], label='lpc estimate')
  plt.plot(f, 20 * np.log10(np.abs(np.fft.fft(y_hat_warp)))[0:N//2], label='wlpc estimate')
  plt.ylabel('magnitude [dB]')
  plt.xlabel('frequency [Hz]')
  plt.legend()
  plt.grid()
  plt.savefig('wlpc_freq.png', dpi=150)
  plt.show()



def plot_ESACF_comp(y_corr, y_rs, y_enh):

  plt.figure(2, figsize=(8,4))
  plt.plot(y_corr[0, :], label='SACF')
  plt.plot(y_rs[0, 0:y_pos.shape[1]], label='streched and cliped')
  plt.plot(y_enh[0, :], label='enhanced')
  plt.ylabel('magnitude')
  plt.xlabel('lag [samples]')
  plt.legend()
  plt.grid()
  plt.savefig('ESACF.png', dpi=150)
  plt.show()


def plot_ESACF(y_enh, onset):

  plt.figure(2, figsize=(8,4))

  for oni in onset:
    plt.plot(y_enh[oni, :], label='ESACF onset'+str(oni))

  plt.ylabel('magnitude')
  plt.xlabel('lag [samples]')
  plt.legend()
  plt.grid()
  plt.savefig('ESACF_mpitch.png', dpi=150)
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
    print("frame length: ", x.shape[1] // hop)

    # gets onsets
    onsets, m = get_onset_mat(file_dir + mat_file_name, var_name)
    print("onsets: ", onsets.shape)


    # --
    # snythesis of channels

    # active channels for synthesis
    active_ch = np.array([1, 1, 1, 1])

    # snythesized
    x_synth = np.zeros(x.shape[1])
    on_synth = np.zeros(onsets.shape[1])

    # snythesis
    for i, ch in enumerate(active_ch==1):

      if ch == True:

        # additive synthesis
        x_synth = x_synth + x[i, :]

        # onset synthesis
        on_synth = np.logical_or(on_synth, onsets[i, :])

    x = x_synth
    onsets = on_synth

    # write file
    #librosa.output.write_wav('./synth.wav', x, fs, norm=False)

    print("x_synth: ", x.shape)
    print("onsets synth: ", onsets.shape)


    # onset deltas to onset samples num
    onset_samples = np.where(onsets == 1)[0] * hop + N / 2
    onset_samples = onset_samples.astype(int)

    # time interval after onset
    t_i = 0.05

    # samples for interval after onset
    N_i = int(t_i * fs)

    print("N_i:", N_i)

    # init onset matrix
    x_of = np.zeros((len(onset_samples), N_i))

    # windowing
    w = np.hanning(N_i)

    # window for onset frames
    for i, on in enumerate(onset_samples):

      # boundary
      if (on + N_i + N // 2) < len(x):

        # onset frame
        x_of[i, :] = x[on+N//2:on+N_i+N//2] * w



    # --
    # pre whitening

    # warped linear predictive coding

    # data = np.array([2, 2, 0, 0, -1, -1, 0, 0, 1, 1])

    # ac_lpc = lpc_corr(data)
    # ac_wlpc = lpc_corr(data, fs=fs, warped=True)

    # print(ac_lpc)
    # print(ac_wlpc)

    # lev = levinson_durbin(ac_lpc, 3)
    # lev_wlpc = levinson_durbin(ac_wlpc, 3)

    # print("lev: ", lev.numerator)
    # print("levw: ", lev_wlpc.numerator)

    lpc_order = 12
    A_lpc = librosa.lpc(x_of[0, :], lpc_order)
    A_wlpc = levinson_durbin( lpc_corr(x_of[0, :], fs=fs, warped=True), lpc_order)

    #print("rosa: ", A_lpc)
    #print("levw: ", A_wlpc.numerator)

    # inverse filtering
    y_hat_rosa = signal.lfilter([0] + -1 * A_lpc[1:], [1], x_of)
    y_hat_warp = signal.lfilter([0] + -1 * np.array(A_wlpc.numerator[1:]), [1], x_of)

    # plot wlpc
    #plot_wlpc(x_of[0, :], y_hat_rosa[0, :], y_hat_warp[0, :], fs)

    print("y_hat_warp: ", y_hat_warp.shape)


    # --
    # channel filtering

    # high and low channel
    n_ch = 2

    # init
    y_ch = np.zeros((n_ch,) + y_hat_warp.shape)

    # high frequency channel
    # high pass params
    order = 4
    f_cut = 1000

    # high pass filter
    b, a = signal.butter(order, f_cut / (fs/2), 'high')
    y_ch[0, :] = signal.filtfilt(b, a, y_hat_warp)

    # half wave rectification
    y_ch[0, :] = y_ch[0, :].clip(min=0)

    # low pass filtering
    b, a = signal.butter(order, f_cut / (fs/2), 'low')
    y_ch[0, :] = signal.filtfilt(b, a, y_ch[0, :])

    # low frequency channel
    # band pass params
    order = 4
    f_cut = np.array([70, 1000])

    # low pass filter
    b, a = signal.butter(order, f_cut / (fs/2), btype='band')
    y_ch[1, :] = signal.filtfilt(b, a, y_hat_warp)
    #plot_filter(b, a, fs)

    print("y_ch: ", y_ch.shape)


    # --
    # Summed Autocorrelation (SACF), done in the frequency domain

    y_corr = np.fft.ifft( np.power(np.abs(np.fft.fft(y_ch[0, :])), 2) + np.power(np.abs(np.fft.fft(y_ch[1, :])), 2))[:, 0:N_i//2]

    print("y_corr: ", y_corr.shape)


    # --
    # ESACF (enhance autocorrelation function)

    # clipping
    y_pos = y_corr.real.clip(min=0)
    print("y_pos: ", y_pos.shape)



    # iteratively enhance
    rs_factors = np.array([2, 3, 4])

    # enhanced variable
    y_enh = y_pos

    # for different time streches
    for rs in rs_factors:

      # resampling variable
      y_rs = np.zeros((y_pos.shape[0], y_enh.shape[1] * rs))

      # for each onset
      for oni in np.arange(len(y_pos)):

        # resample
        y_rs[oni, :] = signal.resample(y_enh[oni, :], y_enh.shape[1] * rs)

      # substract streched signal
      y_enh = (y_enh - y_rs[:, 0:y_pos.shape[1]]).clip(min=0)

    print("y_enh: ", y_enh.shape)


    # --
    # Pitch detection

    # estimated frequency and peaks
    f_est = np.zeros((len(onset_samples), 4))
    peaks = np.zeros((len(onset_samples), 4))

    # find peaks:
    for oni in np.arange(len(onset_samples)):

      # max peak
      h_max = max(y_enh[oni, :])

      # find peaks
      sample_lim = 350
      p, v = signal.find_peaks(y_enh[oni, 0:sample_lim], height=(0.2 * h_max, h_max))

      if not len(v['peak_heights']) == 0:

        # sort for highest peaks
        v_max_idx = np.argsort(v['peak_heights'])[::-1]

        # add most prominent peaks to peaks list
        for i, vi in enumerate(v_max_idx):

          # end if four are found
          if i >= peaks.shape[1]:
            break

          # limit of enhancment is reached
          if i >= 1 and p[vi] >= rs_factors[-1] * peaks[oni, 0]:
            break

          # set peak
          peaks[oni, i] = p[vi]

        # safety for div by zero
        peaks[oni, :][peaks[oni, :] == 0] = 'nan'

        # frequency estimate
        f_est[oni, :] = np.around(fs / peaks[oni, :], decimals=1)

    print('peaks: ', peaks[0, :])
    print('f_est: ', f_est[0, :])


    # plot
    #plot_ESACF_comp(y_corr, y_rs, y_enh)

    #plot_ESACF(y_enh, range(1))


    plt.figure(3, figsize=(8, 4))

    # time vector
    T = 0.01
    t = np.arange(0.023, 0.023 + T * m.shape[1], T)

    # colors mixing
    c1 = 'green'
    c2 = 'orange'

    c_mix = np.linspace(0, 1, 4)
    #a_mix = np.linspace(1, 1, )

    for i, ch in enumerate(active_ch==1):

      # coloring
      c = color_fader(c1, c2, c_mix[i])

      # check if channel is active
      if ch == True:

        # plot midi notes
        plt.plot(t, m[i, :], color=c, label='voice'+str(i+1))


    for i, ons in enumerate(onset_samples):

      # over all estimated frequs
      for j, f in enumerate(f_est[i, :]):

        # coloring
        c = color_fader(c1, c2, c_mix[j])

        plt.scatter(ons * 1/fs - 1/fs, f2midi(f), color=c)

    plt.ylabel('midi notes')
    plt.xlabel('time [s]')
    plt.legend()
    plt.grid()
    plt.savefig('midi' + str(np.sum(active_ch==1)) + '.png', dpi=150)
    plt.show()

    
















