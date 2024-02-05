# --
# dcase 2023
# task 5: Few-shot Bioacoustic Event Detection

import numpy as np
import yaml
import csv
import soundfile as sf

from glob import glob
from audio_file_reader import AudioFileReader


if __name__ == '__main__':
  """
  main
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # reader
  wav_reader = AudioFileReader(cfg['audio_file_reader'])

  # save
  #wav_reader.save_data()

  # read stuff
  for x, y in iter(wav_reader):
    print("audio: ", x)
    print("anno_file: ", y)
    break