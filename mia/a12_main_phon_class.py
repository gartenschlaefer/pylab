# --
# phoneme classification

import numpy as np
import matplotlib.pyplot as plt

# some lean tools
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix

# my personal mia lib
from mia import *
from get_phonemlabels import get_phonemlabels




# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'

  # lda mat files
  file_names = ['A0101B.wav']

  # annotation file
  anno_file = "A0101_Phonems.txt"

  # run through all files
  for file_name in file_names:

    # load file
    x, fs = librosa.load(file_dir + file_name, sr=22050, mono=True)

    # sample params
    N = 1024
    ol = N // 2
    hop = N - ol

    # get phonems
    phonems = get_phonemlabels(file_dir + anno_file)

    # just print them
    #for i, phone in enumerate(phonems):
    #  print("phone: ", i, phone)

    # extract code
    consonant_code_list = np.array(['-', 'D', 'N', 'Q', 'S', 'T', 'Z', 'b', 'd', 'dZ', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'tS', 'v', 'w', 'z'])
    vowel_code_list = np.array(['I', 'E', '{', 'A', 'V', 'U', 'i', 'e', 'e@', 'u', 'o', 'O:', 'aI', 'OI', 'aU', '3`', '@', '@`', 'u:', 'eI', '3:', 'i:', 'A:', '@U'])
    pause_code_list = np.array(['_'])
    end_code = np.array(['-'])

    # get phon_samples
    phon_samples = (np.array([phonems[:, 0], phonems[:, 1]]).astype(float) * fs).astype(int)
    print("phon_samples: ", phon_samples.shape)

    # get labels
    labels = phonems[:, 2]

    # replace it by label codes
    labels[np.isin(labels, pause_code_list)] = 'P'
    labels[np.isin(labels, consonant_code_list)] = 'C'
    labels[np.isin(labels, vowel_code_list)] = 'V'

    print(labels)


    zcr = np.zeros(len(labels))
    rms = np.zeros(len(labels))
    qff = np.zeros(len(labels))

    # run through each phonem save its features
    for i, label in enumerate(labels):

      # get
      x_i = x[phon_samples[0, i] : phon_samples[1, i]]

      print("shape: ", x_i.shape)
      print("label: ", label)

      # window
      w = np.hanning(len(x_i))

      # calculate zcr feature
      zcr[i] = zero_crossing_rate(x_i, w, axis=0)

      # calculate root mean
      rms[i] = calc_rms(x_i, w)

      # first formant
      qff[i] = q_first_formant(x_i, w, fs, f_roi=[300, 1000])


      #print(zcr[i])


    #print(zcr[labels=='P'])
    #print(zcr[labels=='C'])
    #print(zcr[labels=='V'])

    plt.figure()
    plt.scatter(zcr[labels=='P'], rms[labels=='P'], label='P')
    plt.scatter(zcr[labels=='C'], rms[labels=='C'], label='C')
    plt.scatter(zcr[labels=='V'], rms[labels=='V'], label='V')

    plt.ylabel('RMS value')
    plt.xlabel('Zero Crossing Rate')
    plt.legend()

    plt.grid()
    plt.show()


    plt.figure()
    plt.scatter(qff[labels=='P'], rms[labels=='P'], label='P')
    plt.scatter(qff[labels=='C'], rms[labels=='C'], label='C')
    plt.scatter(qff[labels=='V'], rms[labels=='V'], label='V')

    plt.ylabel('RMS value')
    plt.xlabel('First Formant')
    plt.legend()

    plt.grid()
    plt.show()

    plt.figure()
    plt.scatter(qff[labels=='P'], zcr[labels=='P'], label='P')
    plt.scatter(qff[labels=='C'], zcr[labels=='C'], label='C')
    plt.scatter(qff[labels=='V'], zcr[labels=='V'], label='V')

    plt.ylabel('Zero Crossing')
    plt.xlabel('First Formant')
    plt.legend()

    plt.grid()
    plt.show()



    # windowing params
    #t_hop = 0.01
    #hop = int(t_hop * fs)
    #ol = N - hop










    






















