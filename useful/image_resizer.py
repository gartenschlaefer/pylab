"""
image resizer
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

from pathlib import Path
from glob import glob


class ImageResizer():
  """
  Image resize class
  """

  def __init__(self, rs_params):

    # resize params
    self.rs_params = rs_params

    # out path init
    if not os.path.isdir(self.rs_params['out_path']): os.makedirs(self.rs_params['out_path'])

    # shape
    self.image_shape = None


  def resize(self, files):
    """
    resize files
    """

    for file in files:

      # read image
      img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

      # store shape
      self.image_shape = img.shape

      # resize
      r_img = cv2.resize(img, tuple([int(x * self.rs_params['scale']) for x in img.shape[::-1]]), interpolation=cv2.INTER_NEAREST)

      # save
      cv2.imwrite("{}{}".format(self.rs_params['out_path'], Path(file).name), r_img, (cv2.IMWRITE_PXM_BINARY, 1))


  def create_video(self):
    """
    create a video
    """

    img_files = sorted(glob(self.rs_params['out_path'] + '*' + self.rs_params['file_ext']))
    img = cv2.imread(img_files[0])

    # store shape
    self.image_shape = img.shape[0:2]
    print(self.image_shape)

    # init video writer
    video = cv2.VideoWriter(self.rs_params['out_path'] + self.rs_params['video_name'], fourcc=cv2.VideoWriter_fourcc(*'MPEG'), fps=self.rs_params['fps'], frameSize=self.image_shape)

    for file in img_files:

      print(file)
      # read image
      img = cv2.imread(file)

      # write
      video.write(img)

    # end
    video.release()


if __name__ == '__main__':
  """
  main
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # files
  files = sorted(glob(cfg['dataset_path'] + '*' + cfg['file_ext']))

  # resizer init
  ir = ImageResizer(cfg['rs_params'])

  # resize files
  #ir.resize(files)

  # create video
  ir.create_video()