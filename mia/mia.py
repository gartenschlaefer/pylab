# --
# library for mia - Music Information retrivAl

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

# librosa
import librosa

# filtering
from scipy.ndimage.filters import convolve
from scipy import signal


def calc_accuracy(y_pred, y_true):
  """
  calculates accuracy
  """

  # amount of labels
  N = len(y_true)

  # check predictions
  correct_pred = np.sum(y_pred == y_true)
  false_pred = N - correct_pred
  acc = correct_pred / N

  return acc, correct_pred, false_pred


def q_first_formant(x, w, fs, f_roi=[300, 1000]):
  """
  calculates the q value of the fist formant
  """

  # fft
  X = np.fft.fft(x * w, n=2048)

  # sample len
  N = len(X)

  # Amplitude
  Y = 2 / N * np.abs(X[0:N//2])

  # sample roi
  s_roi = ((N / 2) / (fs / 2) * np.array(f_roi)).astype(int)

  # max height
  h_max = max(Y[s_roi[0] : s_roi[1]])

  # find peak
  p, v = signal.find_peaks(Y[s_roi[0] : s_roi[1]], height=(0.3 * h_max, h_max))

  f = np.linspace(0, fs/2, N//2)

  
  if not len(p) == 0:
    #print(p)
    p = p + s_roi[0]

  else:
    p = np.append(p, s_roi[0]+1)
    #print("append: ", p)

  # get first peak
  # plt.figure()
  # plt.plot(f, Y)

  #  plt.scatter(p[0] * (fs / 2) / (N / 2), Y[p[0]])

  # plt.xlim(f_roi)
  # plt.show()

  return Y[p[0]]



def lda_classifier(x, y, method='class_dependent', n_lda_dim=1):
  """
  compute lda classifier
  """

  # n samples, m features
  n, m = x.shape

  # amount of classes
  labels = np.unique(y)
  n_classes = len(labels)

  # averall mean
  mu = np.mean(x, axis=0)

  print("mu: ", mu.shape)

  # class occurence probability
  p_k = np.zeros(n_classes)
  mu_k = np.zeros((n_classes, m))
  cov_k = np.zeros((n_classes, m, m))

  label_list = []

  # calculate mean class occurence and mean vector
  for k, label in enumerate(labels):

    # append label
    label_list.append(label)

    # get class samples
    class_samples = x[y==label, :]

    # class ocurrence probability
    p_k[k] = len(class_samples) / n

    # mean vector of classes
    mu_k[k] = np.mean(class_samples, axis=0)

    # covariance vector of classes
    cov_k[k] = np.cov(class_samples, rowvar=False)

  # calculate between class scatter matrix -> S_b
  S_b = p_k * (mu_k - mu).T @ (mu_k - mu)

  # copy covarianc matrix
  cov_copy = np.copy(cov_k)

  for i in range(len(p_k)):
    cov_copy[i] *= p_k[i]

  # calculate within class scatter matrix -> S_w
  S_w = np.sum(cov_copy, axis=0)


  # class dependent use covariance
  if method == 'class_dependent':

    # init
    w = np.zeros((n_classes, m, n_lda_dim))
    bias = np.zeros(n_classes)
    x_h = np.zeros((n, n_lda_dim))

    # run through all classes
    for k in range(n_classes):

      # compute eigenvector
      eig_val, eig_vec = np.linalg.eig(np.linalg.inv(cov_k[k]) @  S_b)

      # use first eigenvector
      w[k] = eig_vec[:, 0:n_lda_dim].real

      # transformierte daten
      x_h[y==label_list[k]] = (w[k].T @ x[y==label_list[k]].T).T

      # bias
      bias[k] = np.mean(x_h[y==label_list[k]])



  # not class dependent use S_w, TODO:
  else:

    # compute eigenvector
    eig_val, eig_vec = np.linalg.eig(np.linalg.inv(S_w) @  S_b)
    
    # first row eigenvector from eigenvalue 0
    w = eig_vec[:, 0].real

    # bias
    #bias = w.T @ mu

  return w, bias, x_h, label_list




def calc_fisher_ration(x, y):
  """
  calculate the fisher ration of each feature and each class
  """

  # n samples, m features
  n, m = x.shape

  # amount of classes
  labels = np.unique(y)
  n_classes = len(labels)

  # compare labels
  compare_label = []

  # get all labels to compare
  for i in range(n_classes - 1):
    for i_s in range(i + 1, n_classes):
      compare_label.append(labels[i] + ' - ' + labels[i_s])

  # init ratio
  r = np.zeros((m, len(compare_label)))

  # all features
  for j in range(m):

    c = 0

    # all class compares
    for i in range(n_classes - 1):

      for i_s in range(i + 1, n_classes):

        r[j, c] = (np.mean(x[y==labels[i], j]) - np.mean(x[y==labels[i_s], j]))**2 / (np.var(x[y==labels[i], j]) + np.var(x[y==labels[i_s], j]) )
        c += 1
  
  return r, compare_label


def calc_pca(x):
  """
  calculate pca of signal, already ordered, n x m (samples x features)
  """

  # eigen stuff -> already sorted
  eig_val, eig_vec = np.linalg.eig(np.cov(x, rowvar=False))

  # pca transformation
  return np.dot(x, eig_vec)


def get_sdm_threshold(sdm):
  """
  compute the sdm threshold for binarized sdms
  """
  
  # lib for otsu threshold
  from skimage import filters

  F = np.zeros(sdm.shape[0]-1) 

  # run through all diagonals except main diagonal which is zero
  for m in range(1, sdm.shape[0]):

    # mean of diagonal
    F[m-1] = np.mean(sdm * np.eye(sdm.shape[0], k=-m))

  # moving average filter
  F_h = F - np.convolve(F, np.ones(50) / 50, mode='same')

  # low pass filtering
  F_t = signal.lfilter(np.array([1, 0, -1]), 1, F_h)

  # get good diagonals with otsu threshold
  y_diagonals = F_t < filters.threshold_otsu(F_t)

  # concatenate diagonals
  g_conc = np.array([])

  # go through all diagonals
  for m, y in enumerate(y_diagonals):

    # choose best diagonals
    if y:

      # get diagonal
      g_y = np.ravel(sdm * np.eye(sdm.shape[0], k=-m-1))

      # smooth diagonal with moving average with 4
      g_y_smooth = np.convolve(g_y[g_y!=0], np.ones(4) / 4, mode='same')

      # concatenate smoothed diagonals
      g_conc = np.concatenate((g_conc, g_y_smooth))

  # get sdm tresh with 20% of smaller values under it
  return np.sort(g_conc)[len(g_conc)//5]


def chroma_sdm_enhancement(sdm):
  """
  chroma enhancement with weird local mean filters proposed by Eronen
  """

  # enhanced sdm
  enh_sdm = np.copy(sdm)

  # copy for convolution
  conv_sdm = np.copy(sdm)


  # create mean kernels, k1 and k4 are diagonal ones -> important
  k1 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 3
  k2 = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 3
  k3 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 3
  k4 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]) / 3
  k5 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]) / 3
  k6 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 3

  kernels = np.array([k1, k2, k3, k4, k5, k6])

  sdm_dir_mean = np.zeros((kernels.shape[0],) + sdm.shape)

  print("sdm_dir: ", sdm_dir_mean.shape)

  # calculate directional means
  for i, k in enumerate(kernels):
    sdm_dir_mean[i, :] = convolve(conv_sdm, k)

  # do the weird stuff
  for i in range(sdm.shape[0]):
    for j in range(sdm.shape[1]):

      # local mean values
      dir_means = sdm_dir_mean[:, i, j]

      # kernel with local min
      k_local_min = np.argmin(dir_means)

      # if diagonals are minimum local mean
      if k_local_min == 0 or k_local_min == 3:

        # add minimum local mean value
        enh_sdm[i, j] += dir_means[k_local_min]

      # else horizontal or vertical are minimum local mean
      else:

        # add largest local mean value
        enh_sdm[i, j] += np.max(dir_means)

  return enh_sdm


def calc_sdm(feat_frames, distance_measure='euclidean'):
  """
  calculate the self-distance matrix from frame feature vectors
  """

  # init
  M = feat_frames.shape[1]
  sdm = np.zeros((M, M))

  # run through each feature
  for i, feat in enumerate(feat_frames.T):

    # compare with each other feature frame
    for j in range(M):
      
      # calculate distance
      sdm[i, j] = np.linalg.norm(feat - feat_frames[:, j])

  return sdm


def calc_chroma(x, fs, hop=512, n_octaves=5, bins_per_octave=36, fmin=65.40639132514966):
  """
  calculate chroma values with constant q-transfrom and tuning of the HPCP
  """

  # ctq
  C = np.abs(librosa.core.cqt(x, sr=fs, hop_length=hop, fmin=fmin, n_bins=bins_per_octave * n_octaves, bins_per_octave=bins_per_octave, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='reflect', res_type=None))

  # calculate HPCP
  hpcp = HPCP(C, n_octaves, bins_per_octave=bins_per_octave)

  # make a histogram of tuning bins
  hist_hpcp = histogram_HPCP(hpcp, bins_per_octave)

  # tuning
  tuned_hpcp = np.roll(hpcp, np.argmax(hist_hpcp), axis=0)

  return filter_HPCP_to_Chroma(tuned_hpcp, bins_per_octave, filter_type='median')


def frame_filter(feature, frames, filter_type='median'):
  """
  Filtering of two consecutive frames, median or mean filter
  """

  # init
  m_feature = np.zeros((feature.shape[0], len(frames)))

  # for each frame
  for i, frame in enumerate(frames):

    # stopp filtering
    if i == len(frames) - 1:
      end_frame = -1

    else:
      end_frame = frames[i+1]

    # average filter
    if filter_type == 'mean':
      m_feature[:, i] = np.mean(feature[:, frame:end_frame], axis=1)

    # median filter
    else:
      m_feature[:, i] = np.median(feature[:, frame:end_frame], axis=1)

  return m_feature


def create_chord_mask(maj7=False, g6=False):

  # dur
  dur_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

  # mol
  mol_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

  # maj7
  maj7_template = np.array([0.75, 0, 0, 0, 0.75, 0, 0, 0.75, 0, 0, 0, 1.2])

  # 6
  g6_template = np.array([0.75, 0, 0, 0, 0.75, 0, 0, 0.75, 0, 1, 0, 0])

  # chroma labels
  chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

  # chord labels
  chord_labels = chroma_labels + [c + "m" for c in chroma_labels]

  # templates concatenated
  chord_templates = np.array([dur_template, mol_template])

  # append maj7
  if maj7 ==  True:
    chord_templates = np.vstack((chord_templates, maj7_template))
    chord_labels = chord_labels + [c + "maj7" for c in chroma_labels]

  # append g6
  if g6 == True:
    chord_templates = np.vstack((chord_templates, g6_template))
    chord_labels = chord_labels + [c + "6" for c in chroma_labels]

  # init mask
  chord_mask = np.empty((0, 12), int)

  # go through all templates
  for chord_template in chord_templates:
    
    # all chroma values
    for c in range(12):

      # add to events
      chord_mask = np.vstack((chord_mask, np.roll(chord_template, c)))

  return chord_mask, chroma_labels, chord_labels


def filter_HPCP_to_Chroma(tuned_hpcp, bins_per_octave, filter_type='mean'):
  """
  filter hpcp bins per chroma to a single chroma value, mean and median filters are possible
  """
  if filter_type == 'mean':
    chroma = np.mean(np.abs(buffer2D(tuned_hpcp, bins_per_octave // 12)), axis=1)

  else:
    chroma = np.median(np.abs(buffer2D(tuned_hpcp, bins_per_octave // 12)), axis=1)

  return chroma


def histogram_HPCP(hpcp, bins_per_octave):
  """
  create histogram of tuning bins over all chroma and frames
  """
  return np.sum(np.sum(np.abs(buffer2D(hpcp, bins_per_octave // 12)), axis=0), axis=1)



def HPCP(C, n_octaves, bins_per_octave=12):
  """
  Harmonic Pitch Class Profile calculated from cqt C
  """
  return np.sum(np.abs(buffer2D(C, bins_per_octave)), axis=0)


def half_wave_rect(x):
  """
  half wave rectification
  """
  return (x + np.abs(x)) / 2


def spectral_flux(X):
  """
  spectral flux for beat detection
  """
  return np.sum(half_wave_rect(np.diff(X)), axis=1)


def adaptive_whitening(X, mu=0.997, r=0.6):
  """
  adaptive whitening
  """
  # num of frames
  n_frames = X.shape[0]
  N = X.shape[1] // 2

  # single band dft
  X_abs = np.abs(X[:, 0:N])

  # peak spectral profile
  P = np.zeros((n_frames, N))

  # first profile
  P[0, :] = np.maximum(X_abs[0, :], r * np.ones(N))

  # run throgh all frames
  for n in range(1, n_frames):

    # other profiles
    P[n, :] = np.maximum(X_abs[n, :], r * np.ones(N), mu * P[n-1, :])


  # whitening
  X_white = np.divide(X[:, 0:N], P)

  return X_white


def color_fader(c1, c2, mix):
  """
  fades colors
  """
  import matplotlib as mpl

  # convert colors to rgb
  c1 = np.array(mpl.colors.to_rgb(c1))
  c2 = np.array(mpl.colors.to_rgb(c2))

  return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)
  

def lpc_corr(x, fs=44100, warped=False):
  
  # usual autocorrelation
  if warped == False:
    return np.correlate(x, x, mode='full')[len(x)-1:]

  # laged stuff
  x_lag = x

  # length
  N = len(x)

  # correlation array
  r = np.zeros(N)

  # allpass coeffs
  b, a = allpass_coeffs(warped_lambda(fs))

  for i in range(N):
    
    # corr with warped lag
    r[i] = np.sum(x * x_lag)

    # warped lag
    x_lag = signal.lfilter(b, a, x_lag)

  return r


def warped_lambda(fs):
  """
  calc lambda for allpass so that warped frequencies resemble human auditory system
  """
  return 1.0674 * np.power(2 / np.pi * np.arctan(0.06583 * fs / 1000), 0.5) - 0.1916


def allpass_coeffs(lam):
  """
  comput allpass coefficients
  """

  # filter coeffs
  b = np.array([-lam, 1])
  a = np.array([1, -lam])

  return b, a


def get_midi_events(file_name):
  """
  midi envents reader from file
  """

  import mido

  mid = mido.MidiFile(file_name)

  midi_events = np.empty((0, 4), float)

  # cumulative time
  cum_time = 0

  for msg in mid:

    # meta info
    if msg.is_meta:
      continue

    # cumulative time add
    cum_time += msg.time

    # add to events
    midi_events = np.vstack((midi_events, (msg.type == 'note_on', msg.note, msg.time, cum_time)))

  return midi_events


def midi2f(m):
  """
  midi to frequency
  """
  return 440 * np.power(2, ((m - 69) / 12))


def f2midi(f):
  """
  frequency to midi quantization
  """
  return np.round(12 * np.log(f / 440) / np.log(2) + 69)


def acf(x):
  """
  simple auto correlation function, but slowly
  """

  # length of signal
  N = len(x)

  # single sided acf
  xm = np.zeros((2 * N, N))

  # padded with zeros
  x_pad = np.pad(x, (N-1, N))

  for i in np.arange(2 * N):
    
    # rotated signal
    xm[i, :] = x_pad[i:i+N]

  # ACF function
  r = np.dot(xm, x) / (2 * N + 1)

  return r


# --
# score for onset detection
def score_onset_detection(onsets, labels, tolerance=0.02, time_interval=()):
  """
  score functions: Precision, Recall and F-measure
  comparison of onsets to actual labels with tolerance in time measure [s]
  params:
    onsets - onsets instances in time space
    labels - label instances in time space
    tolerance - tolerance between target onset and actual label

  return:
    (P, R, F) - (Precision, Recall, F-Measure)
  """

  # get all onsets in between time interval
  if time_interval:
    onsets = onsets[onsets > time_interval[0]]
    onsets = onsets[onsets < time_interval[1]]
    labels = labels[labels > time_interval[0]]
    labels = labels[labels < time_interval[1]]

  # totals:
  total_onsets = len(onsets)
  total_labels = len(labels)

  # tolerance band of each label
  neg_label_tolerance = labels - tolerance
  pos_label_tolerance = labels + tolerance

  # hits of onsets
  hit_onsets = np.zeros(total_onsets)

  # measure hits of onsets
  for i, onset in enumerate(onsets):

    # count onset hits
    hit_onsets[i] = np.sum(np.logical_and(neg_label_tolerance < onset, pos_label_tolerance > onset))

  # true positives, hit label
  TP = sum(hit_onsets == 1)

  # false positives, false alarm
  FP = sum(hit_onsets == 0)

  # hits of labels
  hit_labels = np.zeros(total_labels)

  # measure hits of labels
  for i, label in enumerate(labels):

    # count label hits
    hit_labels[i] = np.sum(np.logical_and(label - tolerance < onsets, label + tolerance > onsets))

  # false negatives, label missed
  FN = sum(hit_labels == 0)

  # precision
  P = TP / (TP + FP) * 100

  # recall
  R = TP / (TP + FN) * 100

  # f-measure
  F = 2 * P * R / (P + R)

  return (P, R, F)

  
def thresholding_onset(x, thresh):
  """
  thresholding for onset events
  params: 
    x - input sequence
    thresh - threshold vector
  """

  # init
  onset = np.zeros(len(x))

  # set to one if over threshold
  onset[x > thresh] = 1

  # get only single onset -> attention edge problems
  onset = onset - np.logical_and(onset, np.roll(onset, 1))

  return onset


def adaptive_threshold(g, H=10, alpha=0.05, beta=1):
  """
  adaptive threshold with sliding window
  """

  # threshold
  thresh = np.zeros(len(g))
  #alpha_thresh = np.zeros(len(g))

  # sliding window
  for i in np.arange(H//2, len(g) - H//2):

    # median thresh
    thresh[i] = np.median(g[i - H//2 : i + H//2])

    # offset 
    #alpha_thresh[i] = alpha * thresh[i]


  # linear mapping
  #thresh = alpha_thresh + beta * thresh
  thresh = alpha * np.max(thresh) + beta * thresh

  #print("max thresh: ", np.max(thresh))


  return thresh


def complex_domain_onset(X, N):
  """
  complex domain approach for onset detection
  params:
    X - fft
    N - window size
  """

  # calculate phase deviation
  d = phase_deviation(X, N)

  # ampl target
  R = np.abs(X[:, 0:N//2])

  # ampl prediction
  R_h = np.roll(R, 1, axis=0)

  # complex measure
  gamma = np.sqrt(np.power(R_h, 2) + np.power(R, 2) - 2 * R_h * R * np.cos(d))

  # clean up first two indices
  gamma[0] = np.zeros(gamma.shape[1])

  # sum all frequency bins
  eta = np.sum(gamma, axis=1)

  return eta


def phase_deviation(X, N):
  """
  phase_deviation of STFT
  """

  # get unwrapped phase
  phi0 = np.unwrap(np.angle(X[:, 0:N//2]))
  phi1 = np.roll(phi0, 1, axis=0)
  phi2 = np.roll(phi0, 2, axis=0)

  # calculate phase derivation
  d = princarg(phi0 - 2 * phi1 + phi2)

  # clean up first two indices
  d[0:2] = np.zeros(d.shape[1])

  # plt.figure(2)
  # plt.plot(np.transpose(abs(X)[100:102, :]))

  # plt.figure(3)
  # plt.plot(d[1000, :])

  # plt.figure(1)
  # plt.plot(np.transpose(phi0[1000, :]), label='ph1')
  # plt.plot(np.transpose(phi1[1000, :]), label='ph2')
  # plt.plot(np.transpose(phi2[1000, :]), label='ph3')
  # plt.legend()
  # plt.show()

  return d


def amplitude_diff(X, N):
  """
  calculate amplitude differences from consecutive frames
  """

  # absolute amplitude
  X = np.abs(X[:, 0:N//2])

  # difference measure
  d = np.sum(X - np.roll(X, 1), 1)

  # clean up first
  d[0] = 0

  return d


def calc_mfcc(x, fs, N=1024, hop=512, n_filter_bands=8):
  """
  mel-frequency cepstral coefficient
  """

  # stft
  X = stft(x, 2*N, hop)

  # weights
  w_f, w_mel, n_bands = mel_band_weights(n_filter_bands, fs, N)

  # energy of fft
  E = np.power(2 / N * np.abs(X[:, 0:N]), 2)

  # sum the weighted energies
  u = np.inner(E, w_f)

  # discrete cosine transform of log
  return dct(np.log(u), n_bands).T


def dct(X, N):
  """
  discrete cosine transform
  """
  
  # transformation matrix
  H = np.cos(np.pi / N * np.outer((np.arange(N) + 0.5), np.arange(N)))

  # transformed signal
  return np.dot(X, H)


def triangle(M, N):
  """
  create a triangle
  """
  return np.concatenate((np.linspace(0, 1, M), np.linspace(1 - 1 / N, 0, N - 1)))


def mel_band_weights(n_bands, fs, N=1024, overlap=0.5):
  """
  mel_band_weights create a weight matrix of triangluar mel band weights for a filter bank.
  This is used to compute MFCC.
  """

  # hop of samples
  hop = N / (n_bands + 1)

  # calculating middle point of triangle
  mel_samples = np.arange(hop, N, hop)
  f_samples = np.round(mel_to_f(mel_samples / N * f_to_mel(fs / 2)) * N / (fs / 2))

  # round mel samples too
  mel_samples = np.round(mel_samples)

  # complicated hop sizes for frequency scale
  hop_f = (f_samples - np.roll(f_samples, +1))
  hop_f[0] = f_samples[0]

  # triangle shape
  tri = triangle(hop, hop+1)

  # weight init
  w_mel = np.zeros((n_bands, N))
  w_f = np.zeros((n_bands, N))

  for mi in range(n_bands):

    # for equidistant mel scale
    w_mel[mi][int(mel_samples[mi])] = 1
    w_mel[mi] = np.convolve(w_mel[mi, :], tri, mode='same')

    # for frequency scale
    w_f[mi, int(f_samples[mi])] = 1
    w_f[mi] = np.convolve(w_f[mi], triangle(hop_f[mi]+1, hop_f[mi]+1), mode='same')

  # print("w_f: ", f_samples.shape)
  # print("w_f: ", w_f.shape)

  # plt.figure(1)
  # plt.plot(w_f.T)
  # plt.show()

  return (w_f, w_mel, n_bands)


def cepstrum(X, N):
  """
  cepstrum
  """

  # transformation matrix
  H = np.exp(1j * 2 * np.pi / N * np.outer(np.arange(N), np.arange(N)))

  # transformed signal
  Ex = np.log( np.power( np.abs(np.dot(X, H)), 2) )

  cep = np.power( np.abs( np.dot(Ex, H) / N ), 2 )

  return cep


def mel_to_f(m):
  """
  mel to frequency
  """
  return 700 * (np.power(10, m / 2595) - 1)


def f_to_mel(f):
  """
  frequency to mel 
  """
  return 2595 * np.log10(1 + f / 700)


def princarg(p):
  """
  principle argument
  """
  return np.mod(p + np.pi, -2 * np.pi) + np.pi


def parabol_interp(X, p):
  """
  parabolic interpolation
  """

  # gama
  gamma = 0.5 * (X[p-1] - X[p+1]) / (X[p-1] - 2 * X[p] + X[p+1]);
  
  # point of zero gradient in parabel
  k = p + gamma;

  # correction
  alpha = (X[p-1] - X[p]) / ( 1 + 2 * gamma);
  beta = X[p] - alpha * np.power(gamma, 2);

  return (alpha, beta, gamma, k)


def inst_f(X, frame, p, R, N, fs):
  """
  instantaneous frequency
  """

  # calculate phases of the peaks between two frames
  phi1 = np.angle(X)[frame][p]
  phi2 = np.angle(X)[frame + 1][p]

  omega_k = 2 * np.pi * p / N
  delta_phi = omega_k * R - princarg(phi2 - phi1 - omega_k * R)

  return delta_phi / (2 * np.pi * R) * fs


def buffer(X, n, ol=0):
  """
  buffer function like in matlab
  """

  # number of samples in window
  n = int(n)

  # overlap
  ol = int(ol)

  # hopsize
  hop = n - ol

  # number of windows
  win_num = (len(X) - n) // hop + 1 

  # remaining samples
  r = int(np.remainder(len(X), hop))
  if r:
    win_num += 1;

  # segments
  windows = np.zeros((win_num, n))

  # segmentation
  for wi in range(0, win_num):

    # remainder
    if wi == win_num - 1 and r:
      windows[wi] = np.concatenate((X[wi * hop :], np.zeros(n - len(X[wi * hop :]))))

    # no remainder
    else:
      windows[wi] = X[wi * hop : (wi * hop) + n]

  return windows


def buffer2D(X, n, ol=0):
  """
  buffer function like in matlab but with 2D
  """

  # number of samples in window
  n = int(n)

  # overlap
  ol = int(ol)

  # hopsize
  hop = n - ol

  # number of windows
  win_num = (X.shape[0] - n) // hop + 1 

  # remaining samples
  r = int(np.remainder(X.shape[0], hop))
  if r:
    win_num += 1;

  # segments
  windows = np.zeros((win_num, n, X.shape[1]), dtype=complex)

  # segmentation
  for wi in range(0, win_num):

    # remainder
    if wi == win_num - 1 and r:
      windows[wi] = np.concatenate((X[wi * hop :], np.zeros((n - X[wi * hop :].shape[0], X.shape[1]))))

    # no remainder
    else:
      windows[wi, :] = X[wi * hop : (wi * hop) + n, :]

  return windows


def calc_rms(x, w):
  """
  rms value with window
  """
  return np.sqrt(np.mean(x**2 * w))


def st_energy(x, w):
  """
  short term energy
  """
  return np.sum( np.multiply(np.power(x, 2), w), 1) / x.shape[1]


def zero_crossing_rate(x, w, axis=1):
  """
  zero crossing rate
  """

  # first sample
  a = np.sign(x)

  # zero handling
  a[a==0] = 1

  # second sample
  b = np.sign(np.roll(x, -1))

  return np.around( np.sum( np.multiply( np.abs(a - b), w ), axis=axis) / 2 )


def stft(x, N=1024, hop=512):
  """
  short time fourier transform
  """
  # windowing
  w = np.hanning(N)

  # apply windows
  x_buff = np.multiply(w, buffer(x, N, N-hop))

  # transformation matrix
  H = np.exp(1j * 2 * np.pi / N * np.outer(np.arange(N), np.arange(N)))

  # transformed signal
  return np.dot(x_buff, H)


