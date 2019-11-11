# --
# library for mia - Music Information retrivAl

import numpy as np

import matplotlib.pyplot as plt


# --
# score for onset detection
def score_onset_detection(onsets, labels, tolerance=0.02):
  """
  score functions: Precision, Recall and F-measure
  comparison of onsets to actual labels with tolerance in time measure [s]
  """
  
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

  # print
  print("total onsets: ", total_onsets)
  print("total labels: ", total_labels)
  print("true positives: ", TP)
  print("false positives: ", FP)
  print("false negatives: ", FN)

  
  
# --
# thresholding
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


# --
# adaptive threshold
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


# -- 
# phase deviation
def complex_domain_onset(X, N):
  """
  complex domain approach for onset detection
  params:
    X - fft
    N - window size
  """

  # calculate phase deviation
  d = phase_deviation(X, N)

  R = np.abs(X[:, 0:N//2])

  R_h = np.roll(R, 1, axis=0)

  gamma = np.sqrt(np.power(R_h, 2) + np.power(R, 2) - 2 * R_h * R * np.cos(d))

  # clean up first two indices
  gamma[0] = np.zeros(gamma.shape[1])

  print("gamma: ", gamma.shape)

  eta = np.sum(gamma, axis=1)
  print("eta: ", eta.shape)

  return eta


# -- 
# phase deviation
def phase_deviation(X, N):
  """
  phase_deviation of STFT
  """

  # get unwrapped phase
  phi0 = np.unwrap(np.angle(X[:, 0:N//2]))
  print("phi size: ", phi0.shape)

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


# -- 
# Amplitude diff
def amplitude_diff(X, N):
  X = np.abs(X[:, 0:N//2])
  return np.sum((np.roll(X, -1) - X)[:-1], 1)


# --
# dct
def dct(X, N):
  
  # transformation matrix
  H = np.cos(np.pi / N * np.outer((np.arange(N) + 0.5), np.arange(N)))

  # transformed signal
  return np.dot(X, H)


# --
# triangle
def triangle(M):

  # create triangle
  return np.concatenate((np.linspace(0, 1, M // 2), np.linspace(1 - 1 / (M // 2), 0, (M // 2) - 1)))


# --
# mel band weights
def mel_band_weights(M, fs, N=1024, ol_rate=0.5):
  """
  mel_band_weights create a weight matrix of triangluar mel band weights for a filter bank.
  This is used to compute MFCC.
  """

  # overlapping stuff
  ol = int(N // M * ol_rate)
  hop = N // M - ol

  # amount of bands
  n_bands = N // hop

  # calculating middle point of triangle
  mel_samples = np.linspace(hop - 1, N - N // n_bands - 1, n_bands - 1)
  f_samples = np.round(mel_to_f(mel_samples / N * f_to_mel(fs / 2)) * N / (fs / 2))

  # complicated hop sizes for frequency scale
  hop_f = (f_samples - np.roll(f_samples, + 1)) + 1
  hop_f[0] = f_samples[0] + 1

  # triangle shape
  tri = triangle(hop * 2)

  # weight init
  w_mel = np.zeros((n_bands, N))
  w_f = np.zeros((n_bands, N))

  for mi in range(n_bands - 1):

    # for equidistant mel scale
    w_mel[mi][int(mel_samples[mi])] = 1
    w_mel[mi] = np.convolve(w_mel[mi, :], tri, mode='same')

    # for frequency scale
    w_f[mi][int(f_samples[mi])] = 1
    w_f[mi] = np.convolve(w_f[mi, :], triangle(hop_f[mi] * 2), mode='same')

  return (w_f, w_mel, n_bands)


# --
# compute cepstrum
def cepstrum(X, N):

  # transformation matrix
  H = np.exp(1j * 2 * np.pi / N * np.outer(np.arange(N), np.arange(N)))

  # transfored signal
  Ex = np.log( np.power( np.abs(np.dot(X, H)), 2) )

  cep = np.power( np.abs( np.dot(Ex, H) / N ), 2 )

  return cep


# --
# mel to frequency
def mel_to_f(m):
  return 700 * (np.power(10, m / 2595) - 1)


# --
# frequency to mel
def f_to_mel(f):
  return 2595 * np.log10(1 + f / 700)


# --
# principle argument
def princarg(p):
  return np.mod(p + np.pi, -2 * np.pi) + np.pi


# --
# parabolic interpolation
def parabol_interp(X, p):

  # gama
  gamma = 0.5 * (X[p-1] - X[p+1]) / (X[p-1] - 2 * X[p] + X[p+1]);
  
  # point of zero gradient in parabel
  k = p + gamma;

  # correction
  alpha = (X[p-1] - X[p]) / ( 1 + 2 * gamma);
  beta = X[p] - alpha * np.power(gamma, 2);

  return (alpha, beta, gamma, k)


# --
# instantaneous frequency
def inst_f(X, frame, p, R, N, fs):

  # calculate phases of the peaks between two frames
  phi1 = np.angle(X)[frame][p]
  phi2 = np.angle(X)[frame + 1][p]

  omega_k = 2 * np.pi * p / N
  delta_phi = omega_k * R - princarg(phi2 - phi1 - omega_k * R)

  return delta_phi / (2 * np.pi * R) * fs


# --
# buffer equivalent
def buffer(X, n, ol=0):

  # number of samples in window
  n = int(n)

  # overlap
  ol = int(ol)

  # hopsize
  hop = n - ol

  # number of windows
  win_num = (len(X) - n) // hop + 1 

  # remeining samples
  r = int(np.remainder(len(X), hop))
  if r:
    win_num += 1;


  # segments
  windows = np.zeros((win_num, n))

  # segmentation
  for wi in range(0, win_num):
    # remainder
    if wi == win_num - 1 and r:
      windows[wi] = np.concatenate((X[wi * hop :], np.zeros(hop - r)))
    # no remainder
    else:
      windows[wi] = X[wi * hop : (wi + 2) * hop]

  return windows


# --
# short term energy
def st_energy(x, w):
  return np.sum( np.multiply(np.power(x, 2), w), 1) / x.shape[1]


# --
# zero crossing rate
def zero_crossing_rate(X, w):

  # first sample
  a = np.sign(X)

  # zero handling
  a[a==0] = 1

  # second sample
  b = np.sign(np.roll(X, -1))

  return np.around( np.sum( np.multiply( np.abs(a - b), w ), 1) / 2 )