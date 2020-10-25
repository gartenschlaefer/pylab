"""
normalizes audio data
"""

import numpy as np
import matplotlib.pyplot as plt

import re
import librosa
import audiofile
import soundfile

from glob import glob
from pydub import effects

from skimage.util import view_as_windows


def median_filtering(x, sample_length=10):
	"""
	filtering
	"""

	print("x: ", x.shape)
	x_win = view_as_windows(x, (sample_length), step=1)

	print("x_win: ", x_win.shape)

	x_filt = np.median(x_win, axis=1)
	print("x_filt: ", x_filt.shape)

	return x_filt


def dynamic_range_compression(x, threshold=-3, ratio=4.0, attack=5.0, release=50.0):
	"""
	dynamic range compression
	"""

	# normalize
	x = librosa.util.normalize(x, norm=np.inf, axis=0, threshold=None, fill=None)

	# linear threshold
	thresh_lin = np.power(10, threshold/10)

	# applying compression
	x[np.abs(x) >= thresh_lin] = ((np.abs(x[np.abs(x) >= thresh_lin]) - thresh_lin) / ratio + thresh_lin) * np.sign(x[np.abs(x) >= thresh_lin])
	#x[np.abs(x) < thresh_lin] = ((thresh_lin - np.abs(x[np.abs(x) < thresh_lin])) / ratio + np.abs(x[np.abs(x) < thresh_lin])) * np.sign(x[np.abs(x) < thresh_lin])

	# normalize
	x = librosa.util.normalize(x, norm=np.inf, axis=0, threshold=None, fill=None)

	return x


def add_dither(x):
  """
  add a dither signal
  """

  # determine abs min value except from zero, for dithering
  try:
    min_val = np.min(np.abs(x[np.abs(x)>0]))
  except:
    print("only zeros in this signal")
    min_val = 1e-4

  # add some dither
  x += np.random.normal(0, 0.5, len(x)) * min_val

  return x


def plot_waveform(x, fs, e=None, hop=None, onset_frames=None, title='none', xlim=None, ylim=None, plot_path=None, name='None'):
  """
  just a simple waveform
  """

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # setup figure
  fig = plt.figure(figsize=(9, 5))
  plt.plot(t, x)

  # energy plot
  if e is not None:
    plt.plot(np.arange(0, len(x)/fs, 1/fs * hop), e)

  # draw onsets
  if onset_frames is not None:
    for onset in frames_to_time(onset_frames, fs, hop):
      plt.axvline(x=float(onset), dashes=(5, 1), color='k')

  plt.title(title)
  plt.ylabel('magnitude')
  plt.xlabel('time [s]')

  if xlim is not None:
    plt.xlim(xlim)

  if ylim is not None:
    plt.ylim(ylim)

  plt.grid()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()


def pydub_dynamic(file, file_ext, out_path, file_name):
	"""
	dynamics by pydub
	"""

	from pydub import AudioSegment
	from pydub import effects

	x = AudioSegment.from_file(file, format=file_ext)

	x = effects.normalize(x, headroom=0.1)

	x = effects.compress_dynamic_range(x, threshold=-3, ratio=4.0, attack=5.0, release=50.0)

	file_handle = x.export(out_path + file_name, format=file_ext)



if __name__ == '__main__':
	"""
	get examples from recordings
	"""

	# path to files
	#in_path = "./ignore/in/"
	in_path = "./ignore/background/"

	out_path = "./ignore/out/"
	out_path = "./ignore/out_back/"
	#file_ext = "ogg"
	file_ext = "wav"

	# reference file
	ref_file = "./ignore/ref.ogg"


	# --
	# params

	# sampling frequency
	fs = 44100

	# factor to ref
	alpha = 0.5


	# --
	# reference

	# load ref
	x, _ = librosa.load(ref_file, sr=fs)

	# energy measure of ref file
	e_ref = np.sum(np.abs(x)**2) / len(x)

	print("\nenergy ref: [{:.4f}]\n".format(e_ref))


	# --
	# process files

	# get files
	files = glob(in_path + '*.' + file_ext)

	# go through all files
	for file in files:

		# extract filename
		file_name = re.findall(r'[\w+ 0-9]+\.'+file_ext, file)[0]

		# print info
		print("filename: [{}]".format(file_name))


		# # read audio from file
		x, _ = librosa.load(file, sr=fs)

		# # add some dither
		# x = add_dither(x)

		# # energy measure of ref file
		# e = np.sum(np.abs(x)**2) / len(x)


		# # dynamic compression
		# #x = dynamic_range_compression(x, threshold=-3, ratio=4.0, attack=5.0, release=50.0)

		# # filtering
		# #x = median_filtering(x, sample_length=10)

		# half value
		x *= 0.5

		# write output

		#audiofile.write(out_path + file_name, x, fs)
		soundfile.write(out_path + file_name, x, fs)

		# plot sound
		#plot_waveform(x, fs, title=file, name=file)
		#plt.show()

		# with soundfile.SoundFile(file, 'r+') as f:
		# 	while f.tell() < f.frames:
		# 		pos = f.tell()
		# 		data = f.read(1024)
		# 		f.seek(pos)
		# 		f.write(data*0.5)

		#pydub_dynamic(file, file_ext, out_path, file_name)