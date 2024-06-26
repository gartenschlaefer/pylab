# --
# hopfield layers from their repo

import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2

from pathlib import Path
from glob import glob
from hflayers import Hopfield
from hopfield import get_data



if __name__ == '__main__':
  """
  main
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # beta
  beta = 1.0

  print(cfg)

  # get data
  train_imgs, retrieve_imgs, retrieve_file_names = get_data(cfg)

  # stored patterns
  Y = train_imgs

  # create their hopfield
  hopfield = Hopfield( 
    scaling=beta,

    # do not project layer input
    state_pattern_as_static=True,
    stored_pattern_as_static=True,
    pattern_projection_as_static=True,

    # do not pre-process layer input
    normalize_stored_pattern=False,
    normalize_stored_pattern_affine=False,
    normalize_state_pattern=False,
    normalize_state_pattern_affine=False,
    normalize_pattern_projection=False,
    normalize_pattern_projection_affine=False,

    # do not post-process layer output
    disable_out_projection=True)

  for R in retrieve_imgs:

    print("R: ", R.shape)
    print("Y: ", Y.shape)
    print("hop: ", hopfield)
    #stop
    
    # hopfield
    z = hopfield((Y.T, R, Y))

    print(z)