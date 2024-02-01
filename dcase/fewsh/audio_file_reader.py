# --
# wav reader

import os
import numpy as np
import re

from file_reader import FileReader

from glob import glob
from pathlib import Path


class AudioFileReader(FileReader):
  """
  reads wav data
  """

  def __init__(self, cfg):
    """
    init
    """

    # super class init
    super().__init__(cfg)

    # print
    [print(f) for f in self.file_dict['train']], [print(f) for f in self.file_dict['val']]


  def filter_files(self):
    """
    file filter (overwrite) -> handed in init
    """

    # filter train files
    self.file_dict['train'] = [f for f in self.file_dict['train'] if Path(f).parent.name in self.cfg['choose_train_classes']]
    self.file_dict['val'] = [f for f in self.file_dict['val'] if Path(f).parent.name in self.cfg['choose_val_classes']]


  def get_npy_files(self):
    """
    get the saved npy files
    """
    return sorted(glob(self.out_path + '*.npy'))


  def init_data_container(self, data):
    """
    init data container
    """

    # overwrite
    data = { 'x': { }, 'y': { } }

    # input data
    data['x'].update({'audio_samples': np.empty(shape=(0), dtype=np.float16, order='C')})

    return data


  def raw_data_processing(self, data):
    """
    raw processing
    """

    # connect
    x = data['x']['audio_samples']

    return data
     

  def save_data_to_npy(self):
    """
    saves the  data in own file format (remove unnecessary data)
    """

    print("\n--\nsave files to npy in: {}".format(self.out_path))

    # file dict
    for k, v in self.file_dict.items():

      # path
      p = Path(self.out_path + k)

      # skip extraction
      if p.is_dir() and not self.cfg['redo_extraction']: continue 

      # make dir
      if not p.is_dir(): os.makedirs(p)

      # files
      for f in v:

        # extract class name
        class_name = Path(f).parent.name
        file_name = Path(f).stem

        # out base name
        out_base_name = class_name + '.' + file_name
        print(file_name)
        print(class_name)
        print(out_base_name)
        stop


      # files
      print(k)
      print(v)





if __name__ == '__main__':
  """
  main
  """
  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # reader
  wav_reader = AudioFileReader(cfg['audio_file_reader'])

  # save
  wav_reader.save_data_to_npy()

  # get files
  print(wav_reader.get_npy_files())