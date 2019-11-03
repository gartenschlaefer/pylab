import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from get_phonemlabels import get_phonemlabels

from mia import *


def bariton():

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

  n_frames = len(x) // hop
  print("n_frames: ", n_frames)
  print("time: ", hop/fs * n_frames)

  # some vectors
  f = np.arange(0, fs/2, fs/N)
  frames = np.linspace(0, hop/fs * n_frames, n_frames)

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

  # peaks, freq and ampl layout with three partial tones
  peaks = np.empty((0, 4), float)
  ampl = np.empty((0, 4), float)
  ampl_est = np.empty((0, 4), float)
  f_est = np.empty((0, 4), float)


  frames_impact = np.zeros(int(n_frames))

  # save all peaks
  for fi in np.arange(Y.shape[0]):

    # find peaks
    p, v = signal.find_peaks(Y[fi], height=(40, 100))

    # check if all three partial tones are included 
    if p.size != 0 and p[0:4].shape[0] == 4:

      # save peaks and amplitudes directly
      peaks = np.vstack((peaks, p[0:4]))
      ampl = np.vstack((ampl, v["peak_heights"][0:4]))

      # parabolic interpolation for amplitude
      ampl_est = np.vstack((ampl_est, parabol_interp(Y[fi], p[0:4])[1]))

      frames_impact[fi] = 1;

      # save instantaneous frequency estimation
      #f_est = np.vstack((f_est, inst_f(X, fi, p[0:4], hop, N, fs)))

  # save frequency estimation to file
  # np.save('f_est', f_est)
  
  # load inst frequency
  f_est = np.load('f_est.npy')

  # remove first entry (bug in prev. code)
  f_est = np.delete(f_est, 0, 0)

  # frequency should be at c1 = 261,626Hz
  print("-- frequency:")
  print("mean peaks f: ", np.mean(peaks, 0) * fs / N)
  print("mean estimated f: ", np.mean(f_est, 0))
  print("-- Amplitude:")
  print("mean peak A: ", np.mean(ampl, 0))
  print("mean estimated A: ", np.mean(ampl_est, 0))


  # --
  # determine modulation params

  # get start frame idx of significant frames
  n_start_frames = np.where(frames_impact == 1)[0][0]
  n_end_frames = n_frames - n_start_frames - peaks.shape[0]

  # interesting times
  frame_cue = np.concatenate((np.zeros(2 * fs//hop), np.ones(1 * fs//hop), np.zeros(n_frames - 3 * fs//hop)));

  # some zero padding of the peaks
  ampl_est = np.pad(ampl_est, [(n_start_frames, n_end_frames), (0, 0)], 'constant', constant_values=0)
  f_est = np.pad(f_est, [(n_start_frames, n_end_frames), (0, 0)], 'constant', constant_values=0)

  # params of modulation
  delta_ampl = (np.max(ampl_est[frame_cue==1], 0) - np.min(ampl_est[frame_cue==1], 0)) / 2
  delta_f = (np.max(f_est[frame_cue==1], 0) - np.min(f_est[frame_cue==1], 0)) / 2

  print("-- Amplitude Modulation:")
  print("delta_ampl: ", delta_ampl)
  print("-- Frequency Modulation:")
  print("delta_f: ", delta_f)



  #print('frame_cues: ', frame_cue.shape)

  # # --
  # # plot amplitude modulation only cues
  # #
  # plt.figure(2, figsize=(8, 4))
  # [f1, f2, f3, f4] = plt.plot(frames[frame_cue==1], ampl_est[frame_cue==1])
  # plt.ylabel('magnitude [dB]')
  # plt.xlabel('time [s]')
  # plt.legend([f1, f2, f3, f4], ["ampl. f0","ampl. f1","ampl. f2","ampl. f3"], loc=1)
  # plt.savefig('amp_mod_est_cue.png', dpi=150)

  # # --
  # # plot frequency modulation
  # #
  # plt.figure(3, figsize=(8, 4))
  # [f1, f2, f3, f4] = plt.plot(frames[frame_cue==1], f_est[frame_cue==1], label='frequency')
  # plt.ylabel('frequency [Hz]')
  # plt.xlabel('time [s]')
  # plt.legend([f1, f2, f3, f4], ["f0","f1","f2","f3"], loc=1)
  # plt.savefig('freq_mod_est_cue.png', dpi=150)

  # plt.show()

  # # --
  # # plot amplitude modulation
  # #
  # plt.figure(2, figsize=(8, 4))
  # [f1, f2, f3, f4] = plt.plot(frames[frames_impact==1], ampl_est)
  # plt.ylabel('magnitude [dB]')
  # plt.xlabel('time [s]')
  # plt.legend([f1, f2, f3, f4], ["ampl. f0","ampl. f1","ampl. f2","ampl. f3"], loc=1)
  # plt.savefig('amp_mod_est.png', dpi=150)

  # # --
  # # plot frequency modulation
  # #
  # plt.figure(3, figsize=(8, 4))
  # [f1, f2, f3, f4] = plt.plot(frames[frames_impact==1], f_est, label='frequency')
  # plt.ylabel('frequency [Hz]')
  # plt.xlabel('time [s]')
  # plt.legend([f1, f2, f3, f4], ["f0","f1","f2","f3"], loc=1)
  # #plt.savefig('freq_mod_est.png', dpi=150)
  
  # plt.show()


  # --
  # plot bariton dft
  #
  # some frame to plot
  # fi = int(n_frames // 2)
  # print("frame at time: ", hop/fs * fi)

  # p, _ = signal.find_peaks(Y[fi], height=(40, 100))

  # plt.figure(1, figsize=(8, 4))
  # #plt.subplot(211)
  # plt.plot(f, Y[fi], label='bariton')
  # plt.scatter(p[0:4] * fs / N, Y[fi][p[0:4]], label='peaks')

  # plt.title('Baritonvokal frame at time: 5.677s')
  # plt.ylabel('magnitude [dB]')
  # plt.xlabel('frequency [Hz]')
  # plt.grid()

  # plt.xlim(0, 5000)
  # plt.legend()
  # plt.savefig('bar_dft.png', dpi=150)
  # plt.show()


  # plot spectogramm
  # f, t, Sxx = signal.spectrogram(x, fs, nperseg=2048, mode='magnitude')
  # plt.pcolormesh(t, f, Sxx)
  # plt.ylabel('Frequency [Hz]')
  # plt.xlabel('Time [sec]')
  # plt.ylim(0, 3000)
  # plt.show()

  # plt.figure(1)
  # plt.specgram(x, 1024, fs)
  # plt.ylim(0, 2000)
  # plt.show()


# --
# test short term energy and zero crossing rate
def test_ste_zcr():

  # sample params
  N = 1024
  ol = N // 2
  hop = N - ol

  # windowing
  w = np.hanning(N)

  # test signal
  fs = 44100
  k = 1
  A = 1
  x = A * np.cos(2 * np.pi * k / N * np.arange(2 * N))

  # some vectors
  t = np.arange(0, len(x)/fs, 1/fs)

  # buffer in frames
  x_buff = buffer(x, N, ol)

  # short term energy
  E = st_energy(x_buff, w)

  # zero crossing rate
  zcr = zero_crossing_rate(x_buff, w)

  # samples of frame middle points
  frames = np.arange(hop, len(x), hop) / fs
  #print(frames)

  # --
  # short term energy and zcr
  
  # plt.figure(2, figsize=(8, 4))
  # plt.plot(t, x, label='cosine')
  # plt.plot(frames, E, label='st energy', marker='o')
  # plt.plot(frames, zcr, label='zcr', marker='o')

  # for fi in frames:
  #   plt.axvline(x=fi, dashes=(1, 1))

  # plt.ylabel('magnitude')
  # plt.xlabel('time [s]')
  # plt.legend()
  # plt.savefig('ste_zcr.png', dpi=150)

  # plt.show()


# --
# Main function
if __name__ == '__main__':

  # --
  # bariton analysis
  #bariton()

  # --
  # low level features

  print("--- low level features ---")

  # test ste and zcr
  #test_ste_zcr()


  # --
  # Speech recording

  file_dir = './ignore/sounds/'
  file_name = 'A0101B.wav'
  fs, x = wavfile.read(file_dir + file_name)

  
  # sample params
  N = 1024
  ol = N // 2
  hop = N - ol

  n_frames = len(x) // hop

  # prints variables
  print("fs: ", fs)
  print("sample len: ", len(x))
  print("time: ", len(x)/fs)
  print("n_frames: ", n_frames)

  # some vectors
  t = np.arange(0, len(x)/fs, 1/fs)

  # windowing
  w = np.hanning(N)

  # buffer in frames
  x_buff = buffer(x, N, ol)

  # short term energy
  E = st_energy(x_buff, w)

  # zero crossing rate
  zcr = zero_crossing_rate(x_buff, w)

  # interesting region in seconds
  t_roi = np.array([0.5, 1.2])

  # handling with samples, times and frames
  t_sample = np.arange(t_roi[0] * fs, t_roi[1] * fs)
  t_sample = t_sample.astype(int)

  frame_roi = t_roi * fs//hop 
  frame_roi = frame_roi.astype(int)

  frames = np.arange(frame_roi[0], frame_roi[1])
  frames = frames.astype(int)

  time_frames = np.arange(t_roi[0] + hop/fs, t_roi[1] + hop/fs, hop/fs) * fs
  time_frames = time_frames.astype(int)
  
  # fphonem label file
  file_path = "./ignore/"
  file_name = "A0101_Phonems.txt"

  # get phonem labels
  phonems = get_phonemlabels(file_path + file_name)

  # --
  # plot audio file section
  
  plt.figure(2, figsize=(8, 4))
  plt.plot(t[t_sample], x[t_sample] / max(x[t_sample]), label='audiofile', linewidth=0.5)

  plt.plot(t[time_frames], E[frames] / max(E[frames]), label='st energy')
  plt.plot(t[time_frames], zcr[frames] / max(zcr[frames]), label='zcr')

  # plot phonems in normed plot
  for ph in phonems:

    # start time limit
    if float(ph[0]) < t_roi[0]: 
      continue; 

    # stop time limit
    if float(ph[0]) > t_roi[1]: 
      break; 

    # draw vertical lines
    plt.axvline(x=float(ph[0]), dashes=(1, 1), color='k')

    # write phone label
    plt.text(x=float(ph[0]), y=0.9, s=ph[2], color='k', fontweight='semibold')


  plt.ylabel('magnitude')
  plt.xlabel('time [s]')
  plt.grid()
  plt.legend()
  #plt.savefig('audio_measures.png', dpi=150)

  plt.show()


  # --
  # plot audio file
  
  # plt.figure(1, figsize=(8, 4))
  # plt.plot(t, x, label='audiofile')
  # plt.ylabel('magnitude')
  # plt.xlabel('time [s]')
  # plt.xlim(t_start, t_end)
  # plt.legend()
  # plt.savefig('audiofile_zoom.png', dpi=150)

  # plt.show()



