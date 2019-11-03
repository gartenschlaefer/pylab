# --
# library for mia - Music Information retrivAl

import numpy as np


#--
# compute cepstrum
def cepstrum(x, N):

  # transformation matrix
  H = np.exp(1j * 2 * np.pi / N * np.outer(np.arange(N), np.arange(N)))

  # transfored signal
  Ex = np.log( np.power( np.abs(np.dot(x, H)), 2) )

  cep = np.power( np.abs( np.dot(Ex, H) / N ), 2 )

  return cep


#--
# mel to frequency
def mel_to_f(m):
  return 700 * (np.exp(m / 1127.01048) - 1)


#--
# mel to frequency
def f_to_mel(f):
  return 1127.01048 * np.log(1 + f / 700)


#--
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

  # f = np.arange(0, fs/2, fs/N)
  # plt.figure(1)
  # plt.plot(f, np.angle(X)[frame][0:512])
  # plt.plot(f, np.angle(X)[frame + 1][0:512])
  # plt.show()
  # print("phi1: ", phi1)
  # print("phi2: ", phi2)

  omega_k = 2 * np.pi * p / N
  delta_phi = omega_k * R - princarg(phi2 - phi1 - omega_k * R)

  return delta_phi / (2 * np.pi * R) * fs


# --
# buffer equivalent
def buffer(X, n, ol = 0):

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
def zero_crossing_rate(x, w):

  # first sample
  a = np.sign(x)

  # zero handling
  a[a==0] = 1

  # second sample
  b = np.sign(np.roll(x, -1))

  return np.around( np.sum( np.multiply( np.abs(a - b), w ), 1) / 2 )