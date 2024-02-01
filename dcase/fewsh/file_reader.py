# --
# file reader base class

import os

from glob import glob
from pathlib import Path


class FileReader():
  """
  reads wav data
  """

  def __init__(self, cfg):
    """
    init
    """

    # config
    self.cfg = cfg

    # folder setup
    self.out_path = self.cfg['out_path'] + self.cfg['out_folder']
    if not os.path.isdir(self.out_path): os.makedirs(self.out_path)

    # file dict
    self.file_dict = {k: sorted(glob(cfg['src_path'] + v + '**/*' + cfg['ext'])) for k, v in cfg['subdirs'].items()}

    # filter files
    self.filter_files()


  def filter_files(self):
    """
    filter files (overwrite)
    """
    pass


  def get_npy_files(self):
    """
    get the saved npy files
    """
    return sorted(glob(self.out_path + '*.npy'))


  def init_data_container(self, data):
    """
    init data container
    """
    return { 'x': { }, 'y': { } }


  def raw_data_processing(self, data):
    """
    raw processing
    """
    pass
     

  def save_data_to_npy(self):
    """
    saves the  data in own file format (remove unnecessary data)
    """
    pass