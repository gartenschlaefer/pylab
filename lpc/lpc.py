# --
# lpc

# config and lib path
import yaml
import sys
sys.path.append(yaml.safe_load(open("./config.yaml"))["mypylib_path"])

import numpy as np
import librosa as lr
import scipy
from file_reader import FileReader
from plotter import Plotter
from fourier import my_dft
from pathlib import Path


if __name__ == '__main__':
  """
  main
  """

  # config
  cfg = yaml.safe_load(open("./config.yaml"))
  print("--\nconfig: {}\n".format(cfg))

  # file reader
  file_reader = FileReader(cfg['file_reader'])
  file_reader.print_files()

  for f in file_reader:

    # load file
    x_raw, fs = lr.load(f)
    t = np.arange(len(x_raw)) / fs

    # prints
    print("file: {}".format(f))
    print("x: {}\nfs: {}".format(len(x_raw), fs))

    # lpc
    a = lr.lpc(x_raw, order=cfg['lpc']['order'], axis=-1)
    #a = -a[1:]
    b = [1]

    # roots
    a_roots = np.roots(a)

    # formants
    a_roots_p = a_roots[a_roots.imag>=0]
    phi = np.angle(a_roots_p)
    freq = phi * fs / (2 * np.pi)
    bw = -(1 / 2) * fs / (2 * np.pi) * np.log(np.abs(a_roots_p))

    # sorting
    si = np.argsort(phi)
    phi = phi[si]
    bw = bw[si]
    freq = freq[si]

    # formant detection
    formants = [f for i, f in enumerate(freq) if (f > 70) and bw[i] < 400]

    # print
    print("lpc coeffs: {}\nroots: {}\nformants: {}".format(a, a_roots, formants))
    stop
    # freqz
    w_imp, h_imp = scipy.signal.freqz(b, a)

    # new a, b
    #b = np.concatenate(([0], -a[1:]))
    #a = [1]

    x_impulse = np.zeros(len(x_raw))
    x_impulse[0] = 1

    #a = librosa.lpc(y, order=2)
    #b = np.hstack([[0], -1 * a[1:]])
    #y_hat = scipy.signal.lfilter(b, [1], y)

    # lpc prediction
    x_hat = scipy.signal.lfilter(b, a, x_raw)
    x_impulse_hat = scipy.signal.lfilter(b, a, x_impulse)

    # define plotter
    #print(np.array([[1], [a_roots]]))
    #stop
    Plotter(plot_type=100, fig_name='zp').plot([[1], a_roots])

    # define new plotter
    plotter = Plotter(plot_type=10, fig_name='lpc_fft')

    # plot
    #plotter.plot([(w_imp, np.abs(h_imp))], name_addon='freqz_<{}>_od-{}'.format(Path(f).stem, cfg['lpc']['order']))

    # div
    div = 1

    # transform signals
    for k, x in {'raw': x_raw, 'lpc': x_hat, 'imp': x_impulse_hat}.items():

      # info
      k += '_<{}>_od-{}'.format(Path(f).stem, cfg['lpc']['order'])

      # dft
      x_tilde = np.fft.fft(x, n=None, axis=-1, norm=None)

      # my dft
      #x_tilde = my_dft(x, K=len(x))

      # spectrum
      x_tilde_abs = np.abs(x_tilde)
      #x_tilde_abs = np.angle(x_tilde)
      #x_tilde_abs = np.real(x_tilde)
      #x_tilde_abs = np.imag(x_tilde)
      x_tilde_abs = x_tilde_abs[:len(x_tilde_abs)//div]

      # freq
      freq = np.arange(len(x_tilde_abs)) / len(x_tilde_abs) * fs / div

      # data
      data = [(freq, x_tilde_abs)]
      #data = [(t, x)]

      # plot
      plotter.plot(data, name_addon=k)