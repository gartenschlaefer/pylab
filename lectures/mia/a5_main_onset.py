# --
# onset detection

import numpy as np
import matplotlib.pyplot as plt
import librosa

from mia import *
from get_annotations import get_annotations
from np_table import np_table

# --
def plot_wavefile(x, fs, name, annotation_file, save=False):


  #anno_text = ['part 1', 'part 2', 'part 3']

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # plot
  plt.figure(1, figsize=(8, 4))
  plt.plot(t, x, label='audiofile', linewidth=1)

  # annotations
  for i, a in enumerate(get_annotations(annotation_file)):

    # draw vertical lines
    plt.axvline(x=float(a), dashes=(1, 1), color='k')

  plt.title(name)
  plt.ylabel('magnitude')
  plt.xlabel('time [s]')
  #plt.xlim((0, 1))

  plt.grid()
  #plt.legend()
  if save:
    plt.savefig(name.split('.')[0] + '.png', dpi=150)
    
  plt.show()


def primitive_strategies(x, x_buff, N, fs, hop, X, c, name):

  # windowing
  w = np.hanning(N)

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # frame vector
  time_frames = (np.arange(0, len(x) - (N / hop - 1) * hop, hop) + N / 2) / fs 

  #(N / hop - 1)

  print("time_frames", time_frames.shape)

  print("x_buff", x_buff.shape)

  # short time energy
  ste = st_energy(x_buff, w)

  print("ste", ste.shape)
  
  # amplitude difference in stfts
  a_diff = amplitude_diff(X, N)

  print("a_diff", a_diff.shape)

  # interesting time plots
  x_roi = [(0, 0.5), (9.5, 10), (18.5, 19)]
  y_roi = [(-1, 1), (-1, 1), (-1, 1)]

  for r in range(len(x_roi)):

    # plot
    plt.figure(2, figsize=(8, 4))
    plt.plot(t, x / max(x), label='audiofile', linewidth=0.5)
    plt.plot(time_frames, a_diff / max(a_diff), label='ampl diff', linewidth=1.5)
    plt.plot(time_frames, ste / max(ste), label='ste', linewidth=1.5)
    plt.plot(time_frames, c / max(c), label='complex', linewidth=1.5)

    plt.title(name)
    plt.ylabel('magnitude')
    plt.xlabel('time [s]')
    plt.xlim(x_roi[r])
    plt.ylim(y_roi[r])

    plt.grid()
    plt.legend()

    plt.savefig('comparison' + str(r) + '.png', dpi=150)
    plt.show()


# --
def awesome_plot(x, fs, N, hop, c, thresh, anno, tolerance, onset_times, file_name):

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # frame vector
  time_frames = (np.arange(0, len(x) - (N / hop - 1) * hop, hop) + N / 2) / fs 


  # interesting time plots
  x_roi = [(0.93, 1.03), (0, 0.5), (9.5, 10), (18.5, 19)]
  y_roi = [(-0.3, 0.3), (-1, 1), (-1, 1), (-1, 1)]

  for r in range(len(x_roi)):

    # plot
    plt.figure(3, figsize=(8, 4))
    plt.plot(t, x / max(x), label='audiofile', linewidth=1)
    plt.plot(time_frames, c / max(c), label='complex', linewidth=1)
    plt.plot(time_frames, thresh / max(c), label='adapt thresh', linewidth=1)

    #plt.plot(time_frames, onset, label='onset', linewidth=1)

    # annotations labels
    for i, a in enumerate(anno):

      # draw vertical lines
      if i == 0: 
        # put labe to legend
        plt.axvline(x=float(a), dashes=(2, 2), color='k', label="hand-labels")
      else:
        plt.axvline(x=float(a), dashes=(2, 2), color='k')


    # annotations targets

    # tolerance band of each label

    neg_label_tolerance = anno - tolerance
    pos_label_tolerance = anno + tolerance
    green_label = False
    red_label = False

    for i, a in enumerate(onset_times):

      # decide if correct or not -> color
      is_tp = np.sum(np.logical_and(neg_label_tolerance < a, pos_label_tolerance > a))

      # draw vertical lines
      if is_tp == 1:

        if green_label == False: 
          # put label
          green_label = True
          plt.axvline(x=float(a), dashes=(5, 1), color='g', label="targets TP")
        else:
          plt.axvline(x=float(a), dashes=(5, 1), color='g')

      else:
        if red_label == False: 
          # put label
          red_label = True
          plt.axvline(x=float(a), dashes=(5, 1), color='r', label="targets FP")
        else:
          plt.axvline(x=float(a), dashes=(5, 1), color='r')


    plt.title(file_name)
    plt.ylabel('magnitude')
    plt.xlabel('time [s]')
    plt.xlim(x_roi[r])
    plt.ylim(y_roi[r])

    plt.grid()
    plt.legend()

    plt.savefig('class' + str(r) + '.png', dpi=150)
    plt.show()



# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'

  file_names = ['megalovania.wav']
  #file_names = ['imperial_march_plastic_trumpet.wav']

  annotation_file_names = ['megalovania.txt']
  #annotation_file_names = ['imperial_march_plastic_trumpet.txt']

  anno_file_parts = 'megalovania_parts.txt'


  # window length
  N = 512

  # windowing params
  ol = int(0.75 * N)
  hop = N - ol

  # run through all files
  for file_name in file_names:

    print("---sound: ", file_name)

    # load file
    x, fs = librosa.load(file_dir + file_name)

    print("x: ", x.shape)
    print("fs: ", fs)
    n_frames = np.ceil(len(x) / hop - (N / hop - 1))
    print("frame length: ", n_frames)

    # plot wavefile
    #plot_wavefile(x, fs, file_name, file_dir + anno_file_parts, save=False)

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

    # phase deviation
    #phase_deviation(X, N)

    # complex domain
    c = complex_domain_onset(X, N)

    # some primitive strategies compared with complex domain
    #primitive_strategies(x, x_buff, N, fs, hop, X, c, file_name)

    # adaptive threshold
    thresh = adaptive_threshold(c, H=5)

    # get onsets from measure and threshold
    onset = thresholding_onset(c, thresh)

    # annotations
    anno = get_annotations(file_dir + annotation_file_names[0])

    # calculate onset times
    onset_times = (onset * np.arange(0, len(onset)) * hop + N / 2) / fs 
    onset_times = onset_times[onset_times > N / 2 / fs]

    # tolerance
    tolerance = 0.02

    # measure in difficult levels:
    part_times = get_annotations(file_dir + anno_file_parts)

    print("part_times", part_times)


    # --
    # scores

    # overall score
    score = score_onset_detection(onset_times, anno, tolerance=tolerance)

    # print performance
    print("---Performance Measure\n")
    print("Precision: [{:.2f}] | Recall: [{:.2f}] | F-measure: [{:.2f}]".format(score[0], score[1], score[2]))

    # list
    score_list = np.array(score)

    for p in range(len(part_times) - 1):
      # score measure
      score = score_onset_detection(onset_times, anno, tolerance=tolerance, time_interval=(part_times[p], part_times[p+1]))
      score_list = np.vstack((score_list, np.array(score)))

      # print performance
      print("Precision: [{:.2f}] | Recall: [{:.2f}] | F-measure: [{:.2f}]".format(score[0], score[1], score[2]))



    # table header
    header = ['Precision', 'Recall', 'F-measure']

    # write results in table
    np_table('results', score_list, header=header)


    # --
    # awesome plot
    #awesome_plot(x, fs, N, hop, c, thresh, anno, tolerance, onset_times, file_name)
















