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


# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'

  # audio file names
  file_names = ['imagine.mp3']

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
    x = x[0:len(x)//2]

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


