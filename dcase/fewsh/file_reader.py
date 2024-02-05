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

    # anno file dict
    self.anno_file_dict = {k: sorted(glob(cfg['src_path'] + v + '**/*' + cfg['anno_ext'])) for k, v in cfg['subdirs'].items()} if 'anno_ext' in self.cfg.keys() else None

    # filter files
    self.filter_files()


  def filter_files(self):
    """
    filter files (overwrite)
    """
    pass


  def raw_data_processing(self, data):
    """
    raw processing
    """
    pass