# --
# chord analysis wit cqt

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


def test_cq():
  """
  Just for testing, from librosa docu
  """

  # Plot one octave of filters in time and frequency
  basis, lengths = librosa.filters.constant_q(22050)

  plt.figure(figsize=(10, 6))
  plt.subplot(2, 1, 1)
  notes = librosa.midi_to_note(np.arange(24, 24 + len(basis)))
  for i, (f, n) in enumerate(zip(basis, notes[:12])):
      f_scale = librosa.util.normalize(f) / 2
      plt.plot(i + f_scale.real)
      plt.plot(i + f_scale.imag, linestyle=':')

  plt.axis('tight')
  plt.yticks(np.arange(len(notes[:12])), notes[:12])
  plt.ylabel('CQ filters')
  plt.title('CQ filters (one octave, time domain)')
  plt.xlabel('Time (samples at 22050 Hz)')
  plt.legend(['Real', 'Imaginary'], frameon=True, framealpha=0.8)

  plt.subplot(2, 1, 2)
  F = np.abs(np.fft.fftn(basis, axes=[-1]))
  # Keep only the positive frequencies
  F = F[:, :(1 + F.shape[1] // 2)]

  librosa.display.specshow(F, x_axis='linear')

  plt.yticks(np.arange(len(notes))[::12], notes[::12])
  plt.ylabel('CQ filters')
  plt.title('CQ filter magnitudes (frequency domain)')
  plt.tight_layout()
  plt.show()


def plot_whole(C, fs, hop, fmin, bins_per_octave):
  """
  plot whole song
  """
  plt.figure(1, figsize=(8, 4))
  librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=fs, hop_length=hop, x_axis='time', y_axis='cqt_note', fmin=fmin, bins_per_octave=bins_per_octave)
  plt.colorbar(format='%+2.0f dB')
  #plt.title('Constant-Q power spectrum')
  plt.tight_layout()
  plt.show()


def plot_intro(hpcp, fs, hop, fmin, bins_per_octave):
  """
  plot hpcp with all 3 tuning bins
  """
  plt.figure(2, figsize=(8, 4))
  librosa.display.specshow(librosa.amplitude_to_db(hpcp, ref=np.max), sr=fs, hop_length=hop, x_axis='time', y_axis='cqt_note', fmin=fmin, bins_per_octave=bins_per_octave)
  plt.colorbar(format='%+2.0f dB')
  #plt.title('Constant-Q power spectrum')
  plt.tight_layout()
  plt.xlim(0, 13)
  plt.show()


def plot_chord_mask(chord_mask, chroma_labels, chord_labels):
  """
  ploting a chord mask image
  """

  fig, ax = plt.subplots(1,1)

  img = ax.imshow(chord_mask.T, cmap='Greys', aspect='equal')#, extent=[0, 12, 0, 12])
  #plt.imshow(np.transpose(np.abs(M)), aspect='auto', extent=[0, len(x)/fs, tatums[0]*1000, tatums[-1]*1000, ])
  plt.ylabel('chroma')
  plt.xlabel('chord')
  #plt.colorbar()

  ax.set_yticks(np.arange(len(chroma_labels)))
  ax.set_yticklabels(chroma_labels)

  ax.set_xticks(np.arange(len(chord_labels)))
  ax.set_xticklabels(chord_labels)

  plt.xticks(fontsize=10, rotation=90)
  plt.yticks(fontsize=10, rotation=0)

  plt.ylim([-0.5, 11.5])
  plt.show()


def plot_chord_spectrum(chord_est, frames, fs, chord_labels, annotation_file, xlim=(0, 13), step=False, text_height=22):
  """
  plotting a chord spectrum: chords over time
  """
  # setup plot
  fig, ax = plt.subplots(1,1)

  if not step:
    plt.plot(librosa.frames_to_time(frames, sr=fs), chord_est)

  else:
    plt.step(librosa.frames_to_time(frames, sr=fs), chord_est, where='post')

  plt.ylabel('chords')
  plt.xlabel('time [s]')

  ax.set_yticks(np.arange(len(chord_labels)))
  ax.set_yticklabels(chord_labels)

  # add anno
  if annotation_file:
    plot_add_anno(annotation_file, text_height=text_height)

  plt.tight_layout()
  plt.grid()
  plt.xlim(xlim)
  plt.show()


# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/sounds/'

  # audio file names
  file_names = ['LetitBe_Ex9_2019.wav']

  # annotation file
  annotation_file = file_dir + 'let_it_be_anno.txt'

  # run through all files
  for file_name in file_names:

    print("---sound: ", file_name)

    # load file
    x, fs = librosa.load(file_dir + file_name, sr=11025, mono=True)

    # windowing params
    hop = 512

    # print some signal stuff
    print("x: ", x.shape)
    print("fs: ", fs)
    print("hop: ", hop)
    print("frame length: ", len(x) // hop)


    # --
    # Constant Q Transform

    n_octaves = 5
    bins_per_octave = 36
    n_bins = bins_per_octave * n_octaves
    fmin = librosa.note_to_hz('C3')

    # ctq
    C = np.abs(librosa.core.cqt(x, sr=fs, hop_length=hop, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='reflect', res_type=None))
    print("C: ", C.shape)
    
    # plot
    #plot_whole(C, fs, hop, fmin, bins_per_octave)


    # --
    # HPCP

    # calculate HPCP
    hpcp = HPCP(C, n_octaves, bins_per_octave=bins_per_octave)
    print("hpcp: ", hpcp.shape)

    # plot hpcp
    #plot_intro(hpcp, fs, hop, fmin, bins_per_octave)


    # --
    # Peak picking and tuning

    # make a histogram of tuning bins
    hist_hpcp = histogram_HPCP(hpcp, bins_per_octave)

    # get tuning bin of max in hist
    tuning_center_bin = np.argmax(hist_hpcp)

    # tuning
    tuned_hpcp = np.roll(hpcp, tuning_center_bin, axis=0)

    #plot_intro(tuned_hpcp, fs, hop, fmin, bins_per_octave)

    
    # --
    # quantised chromagram

    chroma = filter_HPCP_to_Chroma(tuned_hpcp, bins_per_octave, filter_type='median')
    
    #plot_intro(chroma, fs, hop, fmin, bins_per_octave=12)


    # --
    # chord masking

    chord_mask, chroma_labels, chord_labels = create_chord_mask(maj7=False, g6=False)

    #plot_chord_mask(chord_mask, chroma_labels, chord_labels)


    # --
    # chord detection

    # measure with chord mask
    chord_measure = np.dot(chord_mask, chroma)

    # simplistic chord estimation of each frame
    chord_est = np.argmax(chord_measure, axis=0)

    # frames
    frames = np.arange(len(chord_est))

    #plot_chord_spectrum(chord_est, frames, fs, chord_labels, annotation_file, text_height=4*12-3)


    # --
    # chord detection with median filter of chroma between onsets or tatums

    # get onsets
    onset_frames = librosa.onset.onset_detect(x, sr=fs)

    # chroma meadian filter between onsets
    m_chroma = frame_filter(chroma, onset_frames)

    # measure with chord mask
    m_chord_measure = np.dot(chord_mask, m_chroma)

    # chord estimation
    m_chord_est = np.argmax(m_chord_measure, axis=0)


    # plot chord spectrum intro
    plot_chord_spectrum(m_chord_est, onset_frames, fs, chord_labels, annotation_file, xlim=(0, 13), step=True, text_height=2*12-2)
    #plot_chord_spectrum(m_chord_est, onset_frames, fs, chord_labels, annotation_file=[], xlim=(0, 60), step=True, text_height=2*12-3)





























    
















