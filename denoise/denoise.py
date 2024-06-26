# --
# lpc

# config and lib path
import yaml
import sys
sys.path.append(yaml.safe_load(open("./config.yaml"))["mypylib_path"])

# imports
import numpy as np
import librosa as lr
import scipy
import noisereduce

# more imports
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

  # params
  div = 1

  # files
  for f in file_reader:

    # load file
    x_raw, fs = lr.load(f)
    t = np.arange(len(x_raw)) / fs

    # reduce noise
    y_raw = noisereduce.reduce_noise(x_raw, fs)

    # prints
    print("file: {}".format(f))
    print("x: {}\nfs: {}".format(len(x_raw), fs))

    # define new plotter
    plotter_1d = Plotter(plot_type=10, fig_name='nr')

    # transform signals
    for name_addon, x in {'raw': x_raw, 'noise_red': y_raw}.items():

      # info
      name_addon += '_<{}>'.format(Path(f).stem)

      # dft
      x_tilde = np.fft.fft(x, n=None, axis=-1, norm=None)

      # spectrum
      x_tilde_abs = np.abs(x_tilde)
      #x_tilde_abs = np.angle(x_tilde)
      #x_tilde_abs = np.real(x_tilde)
      #x_tilde_abs = np.imag(x_tilde)
      x_tilde_abs = x_tilde_abs[:len(x_tilde_abs)//div]

      # freq
      freq = np.arange(len(x_tilde_abs)) / len(x_tilde_abs) * fs / div

      # data
      #data = [(freq, x_tilde_abs)]
      data = [(t, x)]

      # plot
      plotter_1d.plot(data, name_addon=name_addon)
