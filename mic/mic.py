"""
mic class
"""

import numpy as np
import queue
import yaml
import os

# sound stuff
import sounddevice as sd
import soundfile
import contextlib

import matplotlib.pyplot as plt


class Mic():
  """
  Mic class
  """

  def __init__(self, mic_params, is_audio_record=False, root_path='./'):

    # arguments
    self.mic_params = mic_params
    self.is_audio_record = is_audio_record
    self.root_path = root_path

    # plot path
    self.plot_path = self.root_path + self.mic_params['plot_path']

    # queue
    self.q = queue.Queue()

    # device
    self.device = sd.default.device[0] if not self.mic_params['select_device'] else self.mic_params['device']
    print("device: ", self.device)

    # determine downsample
    self.downsample = 1

    # get input devices
    self.input_dev_dict = self.extract_devices()

    # show devices
    print("\ndevice list: \n", sd.query_devices())
    print("\ninput device ids: ", self.input_dev_dict.keys())

    # energy threshold in lin scale
    self.energy_thresh = 10**(self.mic_params['energy_thresh_db'] / 10)

    # stream
    self.stream = contextlib.nullcontext()

    # change device flag
    self.change_device_flag = False

    # steam active
    self.stream_active = False

    # audio capture file
    self.audio_record_file = './ignore/capture/audio_record.txt'

    # delete audio capture file
    if os.path.isfile(self.audio_record_file): os.remove(self.audio_record_file)

    # mic callback function
    self.mic_stream_callback = self.callback_mic if not self.is_audio_record else self.callback_mic_record

    # actual fs
    self.fs_actual = None


  def init_stream(self, enable_stream=True):
    """
    init stream
    """

    # set default device
    sd.default.device = self.device


    # exception according to samplerates
    found_fs = False
    while not found_fs:

      for fs in self.mic_params['fs_device']:

        try:
          # init stream
          sd.default.samplerate = self.mic_params['fs_device']
          self.stream = sd.InputStream(device=self.device, samplerate=fs, blocksize=int(self.mic_params['blocksize'] * self.downsample), channels=self.mic_params['channels'], callback=self.mic_stream_callback) if enable_stream else contextlib.nullcontext()
          self.fs_actual = fs
          found_fs = True

        except:
          # fs cannot be used
          print("fs: {} cannot be used!".format(fs))

    print("actual fs: ", self.fs_actual)

    # flags
    self.change_device_flag = False
    self.stream_active = True if enable_stream else False

    print("\ndevice list: \n", sd.query_devices())


  def change_device(self, device):
    """
    change to device
    """
    self.change_device_flag = True
    self.device = device
    print("changed device: ", device)


  def change_energy_thresh_db(self, e):
    """
    change energy threshold
    """
    self.mic_params['energy_thresh_db'] = e
    self.energy_thresh = 10**(e / 10)
    print("changed energy thresh: ", e)


  def extract_devices(self):
    """
    extract only input devices
    """
    return {i:dev for i, dev in enumerate(sd.query_devices()) if dev['max_input_channels']}


  def callback_mic(self, indata, frames, t, status):
    """
    input stream callback
    """
    self.q.put(indata[::self.downsample, 0].copy())


  def callback_mic_record(self, indata, frames, t, status):
    """
    input stream callback with record
    """

    # primitive downsampling
    chunk = indata[::self.downsample, 0].copy()

    # add to queue with primitive downsampling
    self.q.put(chunk)
    
    # write record to file
    #with open(self.audio_record_file, 'a') as f: [f.write('{:.5e},'.format(i)) for i in chunk]


  def clear_mic_queue(self):
    """
    clear the queue after classification
    """

    # process data
    if self.q.qsize():

      # init
      x_collect = np.empty(shape=(0), dtype=np.float32)
      e_collect = np.empty(shape=(0), dtype=np.float32)

      # process data
      for i in range(self.q.qsize()):

        # get chunk
        x = self.q.get()

        # append chunk
        x_collect = np.append(x_collect, x.copy())

        # append energy level
        e_collect = np.append(e_collect, 1)

      # detect onset
      e_onset, is_onset = self.onset_energy_level(x_collect, alpha=self.energy_thresh)


  def read_mic_data(self):
    """
    reads the input from the queue
    """

    # process data
    if self.q.qsize():

      # init
      x_collect = np.empty(shape=(0), dtype=np.float32)
      e_collect = np.empty(shape=(0), dtype=np.float32)

      for i in range(self.q.qsize()):

        # get data
        x = self.q.get()

        # append chunk
        x_collect = np.append(x_collect, x.copy())

        # append energy level
        e_collect = np.append(e_collect, 1)

      # detect onset
      e_onset, is_onset = self.onset_energy_level(x_collect, alpha=self.energy_thresh)

      return is_onset

    return False


  def onset_energy_level(self, x, alpha=0.01):
    """
    onset detection with energy level, x: [n]
    """

    # energy calculation
    e = x.T @ x / len(x)

    return e, e > alpha


  def save_audio_file(self, file, x):
    """
    saves collection to audio file
    """

    # has not recorded audio
    if not self.is_audio_record:
      print("***you did not set the record flag!")
      return

    # save audio
    soundfile.write(file, x, self.fs_actual, subtype=None, endian=None, format=None, closefd=True)


  def stop_mic_condition(self, time_duration):
    """
    stop mic if time duration is exceeded (memory issue in recording)
    """

    return (self.q.qsize() * self.mic_params['blocksize'] >= (time_duration * self.fs_actual)) and self.is_audio_record


  def read_mic_queue(self):
    """
    read queue
    """

    # read data
    x = np.empty(shape=(0), dtype=np.float32)

    # read out elements
    while not mic.q.empty(): x = np.append(x, mic.q.get_nowait())

    return x


def plot_waveform(x, name='', plot_path=None,  show_plot=False, close_plot=True):
  """
  plot mfcc extracted features only (no time series)
  """

  # setup figure
  fig = plt.figure()

  # create axis
  ax = plt.axes()

  # plot selected mfcc
  im = ax.plot(x)

  # axis off
  #plt.axis('off'), ax.axis('off')

  # tight plot
  plt.tight_layout()

  # plot the fig
  if plot_path is not None: plt.savefig(plot_path + name + '.png', dpi=100)
  if show_plot: plt.show()
  if close_plot: plt.close()


if __name__ == '__main__':
  """
  mic
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # create mic instance
  mic = Mic(mic_params=cfg['mic_params'], is_audio_record=True)

  # init stream
  mic.init_stream()

  
  # stream and update
  with mic.stream:

    print("recording...")
    while not mic.stop_mic_condition(time_duration=1):
      pass

  # read data
  x = mic.read_mic_queue()

  # plot waveform
  plot_waveform(x, name='None', plot_path=None,  show_plot=True)

  # save audio
  mic.save_audio_file('./out_record.wav', x)
