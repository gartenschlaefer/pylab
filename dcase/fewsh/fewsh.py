# --
# dcase 2023
# task 5: Few-shot Bioacoustic Event Detection

import numpy as np
import yaml

from glob import glob


if __name__ == '__main__':
  """
  main
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  print(cfg)

