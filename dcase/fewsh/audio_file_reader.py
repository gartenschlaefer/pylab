# --
# wav reader

import os
import numpy as np
import re
import soundfile as sf
import csv

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

    #print(self.file_dict)
    #stop

    # print
    [print(f) for f in self.file_dict['train']], [print(f) for f in self.file_dict['val']]
    [print(f) for f in self.anno_file_dict['train']], [print(f) for f in self.anno_file_dict['val']]


  def __iter__(self):
    """
    iterate
    """
    self.it_audio_files = [Path(f) for f in self.get_audio_files()]
    self.it_anno_files = [Path(f).parent / (str(Path(f).stem) + '.csv') for f in self.it_audio_files]
    self.it = 0
    self.n_it = len(self.it_audio_files)
    return self


  def __next__(self):
    """
    next
    """

    # stop condition
    if self.it >= self.n_it: raise StopIteration

    # extracting
    x, y = self.iter_file_extraction(self.it_audio_files[self.it], self.it_anno_files[self.it])

    # iterate
    self.it += 1

    return (x, y)


  def iter_file_extraction(self, audio_file, anno_file):
    """
    file extraction
    """

    # assert file naming
    assert audio_file.stem == anno_file.stem

    # get split
    split_count = int(re.sub('s-', '', audio_file.stem))

    # read
    x_raw, fs = sf.read(str(audio_file))

    # audio data
    x = {'samples': x_raw, 'fs': fs, 'start_sec': split_count * self.cfg['split_in_sec'], 'dataset_id': audio_file.parent.name, 'file': audio_file}

    # parse dict
    anno_data = []

    # read lines
    with open(anno_file, mode='r')as af:
      
      # each line
      for line in csv.DictReader(af):
        anno_data.append(line)

    # labels
    y = {'anno': anno_data, 'dataset_id': anno_file.parent.name, 'file': anno_file}

    return (x, y)


  def filter_files(self):
    """
    file filter (overwrite) -> handled in init
    """

    # filter files
    self.file_dict['train'] = [f for f in self.file_dict['train'] if Path(f).parent.name in self.cfg['choose_train_classes']]
    self.file_dict['val'] = [f for f in self.file_dict['val'] if Path(f).parent.name in self.cfg['choose_val_classes']]
    self.anno_file_dict['train'] = [f for f in self.anno_file_dict['train'] if Path(f).parent.name in self.cfg['choose_train_classes']]
    self.anno_file_dict['val'] = [f for f in self.anno_file_dict['val'] if Path(f).parent.name in self.cfg['choose_val_classes']]


  def get_npy_files(self):
    """
    get the saved npy files
    """
    return sorted(glob(self.out_path + '**/*' + '.npy', recursive=True))


  def get_audio_files(self):
    """
    get the saved npy files
    """
    return sorted(glob(self.out_path + '**/*' + self.cfg['ext'], recursive=True))


  def raw_data_processing(self, data):
    """
    raw processing
    """
    return data
     

  def save_data(self):
    """
    saves the  data in own file format (remove unnecessary data)
    """

    print("\n--\nsave files in: {}".format(self.out_path))

    # file dict
    for k, v in self.file_dict.items():

      # path
      p = Path(self.out_path + k)

      # make dir
      if not p.is_dir(): os.makedirs(p)

      # files
      for f in v:

        # extract class name
        class_name = Path(f).parent.name
        file_name = Path(f).stem

        # get corresponding anno file
        anno_file = Path(f).parent / (str(Path(f).stem) + '.csv')

        # assert existance
        assert anno_file.is_file()

        # parse dict
        anno_data = []

        # read lines
        with open(anno_file, mode='r')as af:
          
          # each line
          for line in csv.DictReader(af):
            anno_data.append(line)

        # out base name
        out_base_name = class_name + '.' + file_name

        # create folder
        new_out_dir = p / out_base_name

        # skip extraction
        if new_out_dir.is_dir() and not self.cfg['redo_extraction']: continue

        # make dir
        if not new_out_dir.is_dir(): os.makedirs(new_out_dir)

        # read file
        x_raw, fs = sf.read(f)

        # samples
        n_samples_split = fs * self.cfg['split_in_sec']
        # add zeros to get 1min files

        # remainder
        remainder_samples = len(x_raw) % n_samples_split

        # padding to split length
        x_raw = np.pad(x_raw, (0, n_samples_split - remainder_samples)) if remainder_samples else x_raw

        # split
        x_split = np.array(np.array_split(x_raw, n_samples_split))
        x_split = x_split.reshape(-1, len(x_split))

        # go through each split
        for i, x in enumerate(x_split):

          # actual start time
          actual_start_time = i * self.cfg['split_in_sec']

          # get specific anno data
          relevant_anno = [d for d in anno_data if float(d['Starttime']) >= actual_start_time and float(d['Starttime']) < actual_start_time + self.cfg['split_in_sec']]

          # processing
          x_proc = self.raw_data_processing(x)

          # file names
          out_file = new_out_dir / 's-{:04}{}'.format(i, self.cfg['ext'])
          out_file_anno = new_out_dir / 's-{:04}{}'.format(i, '.csv')

          # write with sf
          sf.write(str(out_file), x_proc, fs, subtype=None, endian=None, format=None, closefd=True)

          # csv
          with open(out_file_anno, 'w') as anno_file:
            csv_writer = csv.DictWriter(anno_file, fieldnames=relevant_anno[0].keys() if len(relevant_anno) else ['None'])
            csv_writer.writeheader()
            csv_writer.writerows(relevant_anno)
          #np.save(out_file, x_proc)


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
  wav_reader.save_data()

  # get files
  #print(wav_reader.get_audio_files())

  # read stuff
  for data in iter(wav_reader):
    print("data: ", data)
    break