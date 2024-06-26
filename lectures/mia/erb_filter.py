import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def get_erb_params():
  """
  ERB scale parameters
  can be changed here for different scaling
  """

  # params
  ear_q = 9.26449
  min_bw = 24.7
  order = 1;

  return (ear_q, min_bw, order)



def erb_filter_bank(x, fs, n_ch, low_freq, high_freq):
  """
  create a gammatone filter bank
  """

  #if not
  # get coefs
  f_coefs, f_c = make_erb_filters(fs, n_ch, low_freq, high_freq)

  # get coeffs
  A0  = f_coefs[0, :]
  A11 = f_coefs[1, :]
  A12 = f_coefs[2, :]
  A13 = f_coefs[3, :]
  A14 = f_coefs[4, :]
  A2  = f_coefs[5, :]
  B0  = f_coefs[6, :]
  B1  = f_coefs[7, :]
  B2  = f_coefs[8, :]
  gain = f_coefs[9, :] 

  # init filter bank
  filter_bank = np.zeros((n_ch,) + x.shape)

  # use coeffs and filter
  for ch in range(n_ch):

    # filter
    y1 = lfilter([A0[ch], A11[ch], A2[ch]] / gain[ch], [B0[ch], B1[ch], B2[ch]], x)
    y2 = lfilter([A0[ch], A12[ch], A2[ch]], [B0[ch], B1[ch], B2[ch]], y1)
    y3 = lfilter([A0[ch], A13[ch], A2[ch]], [B0[ch], B1[ch], B2[ch]], y2)
    y4 = lfilter([A0[ch], A14[ch], A2[ch]], [B0[ch], B1[ch], B2[ch]], y3)

    # add to filter bank
    filter_bank[ch, :] = y4

  return (filter_bank, f_c)




def make_erb_filters(fs, n_ch, low_freq, high_freq):
  """
  make_erb_filters(fs, n_ch, low_freq)
  computes filter coeffs for gammatone filters
  params:
    fs - Sampling rate
    n_ch - number of equally spaced channels, or vector of center frequencies
    low_freq - lowest frequency
  """

  # sampling time
  T = 1 / fs

  # calculate center frequencies
  f_c = erb_space(n_ch, low_freq, high_freq)

  # get erb params
  ear_q, min_bw, order = get_erb_params()

  # some erb stuff
  erb = np.power( np.power(f_c / ear_q, order) + np.power(min_bw, order), 1 / order )
  B = 1.019 * 2 * np.pi * erb

  # coeffs
  A0 = T
  A2 = 0
  B0 = 1
  B1 = -2*np.cos(2*f_c*np.pi*T) / np.exp(B*T)
  B2 = np.exp(-2*B*T);

  A11 = -(2*T*np.cos(2*f_c*np.pi*T) / np.exp(B*T) + 2*np.sqrt(3+2**1.5)*T*np.sin(2*f_c*np.pi*T) / np.exp(B*T))/2;
  A12 = -(2*T*np.cos(2*f_c*np.pi*T) / np.exp(B*T) - 2*np.sqrt(3+2**1.5)*T*np.sin(2*f_c*np.pi*T) / np.exp(B*T))/2;
  A13 = -(2*T*np.cos(2*f_c*np.pi*T) / np.exp(B*T) + 2*np.sqrt(3-2**1.5)*T*np.sin(2*f_c*np.pi*T) / np.exp(B*T))/2;
  A14 = -(2*T*np.cos(2*f_c*np.pi*T) / np.exp(B*T) - 2*np.sqrt(3-2**1.5)*T*np.sin(2*f_c*np.pi*T) / np.exp(B*T))/2;

  # gain
  gain = np.abs( 
    (-2*np.exp(4j*f_c*np.pi*T)*T + 2*np.exp(-(B*T) + 2j*f_c*np.pi*T) * T * (np.cos(2*f_c*np.pi*T) - np.sqrt(3 - 2**(3/2)) * np.sin(2*f_c*np.pi*T))) * 
    (-2*np.exp(4j*f_c*np.pi*T)*T + 2*np.exp(-(B*T) + 2j*f_c*np.pi*T) * T * (np.cos(2*f_c*np.pi*T) + np.sqrt(3 - 2**(3/2)) * np.sin(2*f_c*np.pi*T))) *
    (-2*np.exp(4j*f_c*np.pi*T)*T + 2*np.exp(-(B*T) + 2j*f_c*np.pi*T) * T * (np.cos(2*f_c*np.pi*T) - np.sqrt(3 + 2**(3/2)) * np.sin(2*f_c*np.pi*T))) *
    (-2*np.exp(4j*f_c*np.pi*T)*T + 2*np.exp(-(B*T) + 2j*f_c*np.pi*T) * T * (np.cos(2*f_c*np.pi*T) + np.sqrt(3 + 2**(3/2)) * np.sin(2*f_c*np.pi*T))) / 
    (np.power((-2 / np.exp(2*B*T) - 2*np.exp(4j*f_c*np.pi*T) +  2*(1 + np.exp(4j*f_c*np.pi*T)) / np.exp(B*T)), 4)))

  # coeffs in array
  f_coefs = np.array([A0 * np.ones(len(f_c)), A11, A12, A13, A14, A2 * np.ones(len(f_c)), B0 * np.ones(len(f_c)), B1, B2, gain])

  return (f_coefs, f_c)




def erb_space(n_ch, low_freq, high_freq):
  """
  equally erb spaces between low and high freq
  """

  # get params
  ear_q, min_bw, order = get_erb_params()

  # calculate center frequencies
  f_c = -(ear_q * min_bw) + np.exp((np.arange(n_ch) + 1) * (-np.log(high_freq + ear_q*min_bw) + np.log(low_freq + ear_q * min_bw)) / n_ch) * (high_freq + ear_q * min_bw)

  return f_c



def plot_filter_bank(x, fs):
  """
  plot filter bank fft
  """

  N = int(x.shape[1])

  # frequency vector
  f = np.arange(0, fs/2, fs/N)

  # plot
  plt.figure(1, figsize=(8, 4))
  Y = 20 * np.log10(np.abs(np.fft.fft(x)))[:, 0:N//2]
  plt.plot(f, np.transpose(Y))

  plt.xscale('log')
  plt.ylabel('magnitude [dB]')
  plt.xlabel('frequency [Hz]')
  plt.ylim((-60, 5))
  plt.grid()
  #plt.savefig('erb_filter_bank.png', dpi=150)
  plt.show()



# --
# Main function
if __name__ == '__main__':

  fs = 22050
  n_ch = 10
  f_low = 100
  f_high = 5000

  # input response
  x = np.zeros(1024)
  x[0] = 1

  fb, fc = erb_filter_bank(x, fs, n_ch, f_low, f_high)

  print("center frequencies: ", fc)
  plot_filter_bank(fb, fs)