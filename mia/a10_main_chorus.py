# --
# chorus detection

import numpy as np
import matplotlib.pyplot as plt

# librosa
import librosa
import librosa.display

# my personal mia lib
from mia import *
from get_annotations import get_annotations, get_annotations_text, plot_add_anno

from get_annotations import plot_add_anno

from scipy import signal
from scipy.io import wavfile


def libroasa_comparance(x, fs):
	"""
	compare with librosa solution of the similarity matrix
	"""
	
	chroma = librosa.feature.chroma_stft(x, sr=fs)

	onset_env = librosa.onset.onset_strength(x, sr=fs)

	tempo, beats = librosa.beat.beat_track(x, sr=fs, onset_envelope=onset_env, trim=False)

	print("tempo: ", tempo)
	print("beats: ", beats.shape)

	sync = librosa.util.sync(chroma, beats)

	chroma_stack = librosa.feature.stack_memory(sync, n_steps=5, mode='edge')

	S = librosa.segment.recurrence_matrix(chroma_stack, sym=True, mode='affinity')

	plt.figure(1)
	librosa.display.specshow(S, cmap='GnBu', x_axis='frames', y_axis='frames')
	plt.colorbar()
	plt.show()


def plot_whole_chroma(C, fs, hop, fmin, bins_per_octave, annotation_file=[]):
  """
  plot whole song
  """
  plt.figure(1, figsize=(8, 4))
  librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=fs, hop_length=hop, x_axis='time', y_axis='chroma', fmin=fmin, bins_per_octave=bins_per_octave)
  plt.colorbar(format='%+2.0f dB')
  #plt.title('Constant-Q power spectrum')

  # add anno
  if annotation_file:
    plot_add_anno(annotation_file, text_height=10)

  plt.tight_layout()
  plt.show()


def plot_mfcc_spec(mfcc, fs, hop, annotation_file=[]):
  """
  plot mfcc spectrum
  """
  print("time: ", mfcc.shape[1] * hop / fs)

  # for ploting issues
  if(mfcc.shape[0] > mfcc.shape[1]):
    mfcc = mfcc.T

  plt.figure(1, figsize=(8, 4))
  plt.imshow(mfcc, aspect='auto', vmin=-20, vmax=40, extent=[0, mfcc.shape[1] * hop / fs, mfcc.shape[0], 0])
  
  plt.ylabel('mfcc band')
  plt.xlabel('time [s]')

  # add anno
  if annotation_file:
    plot_add_anno(annotation_file, text_height=10)
  
  plt.colorbar()
  plt.tight_layout()
  plt.show()


# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'

  # audio file names
  file_names = ['imagine.mp3']

  anno_file = 'imagine_anno.txt'

  # run through all files
  for file_name in file_names:

    print("---sound: ", file_name)

    # load file
    #x, fs = librosa.load(file_dir + file_name, mono=True)
    #np.save('x', x)
    #np.save('fs', fs)
    x = np.load('x.npy')
    fs = np.load('fs.npy')

    # debug -> faster
    x = x[0:len(x)]

    # windowing params
    N = 1024
    hop = 512
    ol = N - hop

    # print some signal stuff
    print("x: ", x.shape)
    print("fs: ", fs)
    print("hop: ", hop)
    print("frame length: ", len(x) // hop)

    #plt.figure(1)
    #plt.plot(x)
    #plt.show()

    #libroasa_comparance(x, fs)

    # --
    # chroma features

    n_octaves = 4
    bins_per_octave = 36
    fmin = librosa.note_to_hz('C3')

    chroma = calc_chroma(x, fs, hop, n_octaves, bins_per_octave, fmin)
    #plot_whole_chroma(chroma, fs, hop, fmin, bins_per_octave=12, annotation_file=anno_file)


    # --
    # mfcc features

    mfcc = calc_mfcc(x, fs, N, hop, n_filter_bands=12)

    # for some nan and inf parts set to -60dB
    mfcc[np.isnan(mfcc)] = -60
    mfcc[np.isinf(mfcc)] = -60

    #mfcc = librosa.feature.mfcc(x, fs, S=None, n_mfcc=12, dct_type=2, norm='ortho', lifter=0)
    #print("mfcc: ", mfcc)
    #plot_mfcc_spec(mfcc, fs, hop, annotation_file=anno_file)




