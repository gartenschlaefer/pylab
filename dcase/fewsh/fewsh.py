# --
# dcase 2023
# task 5: Few-shot Bioacoustic Event Detection

import numpy as np
import yaml
import csv

from glob import glob


def get_dataset_classes(cfg):

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

  return train_class_dict, val_class_dict


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


  #print(train_class_dict)
  print(val_class_dict)

