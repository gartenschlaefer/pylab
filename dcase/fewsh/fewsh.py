# --
# dcase 2023
# task 5: Few-shot Bioacoustic Event Detection

import numpy as np
import yaml
import csv
import soundfile as sf

from glob import glob
from audio_file_reader import AudioFileReader


def get_dataset_classes(cfg):
  """
  get class dict
  """

  # csv files
  class_anno_csv_files = [cfg['dataset']['root_path'] + f for f in cfg['dataset']['class_anno_csv_files']]

  # anno list
  anno_list = []

  # class dict
  train_class_dict = { k: {} for k in cfg['dataset']['choose_train_classes']}
  val_class_dict = { k: {} for k in cfg['dataset']['choose_val_classes']}

  # read them
  for file in class_anno_csv_files:

    with open(file) as f:
      anno_list += list(csv.DictReader(f))

  # add classes
  [train_class_dict[entry['dataset']].update({entry['class_code']: entry['class_name']}) for entry in anno_list if entry['dataset'] in train_class_dict.keys()]
  [val_class_dict[entry['dataset']].update({entry['recording']: entry['class_name']}) for entry in anno_list if entry['dataset'] in val_class_dict.keys()]

  # print infos
  #print(train_class_dict)
  #print(val_class_dict)

  return train_class_dict, val_class_dict


def get_train_data_files(cfg):
  """
  get
  """

  # get data files
  data_file_dict = {'{}'.format(c): sorted(glob(cfg['dataset']['root_path'] + cfg['dataset']['subdir_tain'] + '{}/*'.format(c) + cfg['dataset']['audio_file_ext'])) for c in cfg['dataset']['choose_train_classes']}
  anno_file_dict = {'{}'.format(c): sorted(glob(cfg['dataset']['root_path'] + cfg['dataset']['subdir_tain'] + '{}/*'.format(c) + cfg['dataset']['anno_file_ext'])) for c in cfg['dataset']['choose_train_classes']}

  return data_file_dict, anno_file_dict


if __name__ == '__main__':
  """
  main
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # init
  train_class_dict, val_class_dict = {}, {}

  # get classes
  train_class_dict, val_class_dict = get_dataset_classes(cfg)

  # get data
  data_file_dict, anno_file_dict = get_train_data_files(cfg)


  # reader
  wav_reader = AudioFileReader(cfg['audio_file_reader'])

  # save
  wav_reader.save_data_to_npy()

  # get files
  print(wav_reader.get_npy_files())
  

  # splitting things up
  # training for all classes
  # for c in cfg['dataset']['choose_train_classes']:

  #   # data file
  #   for data_file, anno_file in zip(data_file_dict[c], anno_file_dict[c]):

  #     print(data_file)
  #     print(anno_file)

  #     # read file
  #     x, fs = sf.read(data_file)
  #     print(fs)
  #     print(x)
  #     print(len(x))

  #     # sample split
  #     n_samples_split = cfg['dataset']['split_in_sec'] * fs
  #     stop
