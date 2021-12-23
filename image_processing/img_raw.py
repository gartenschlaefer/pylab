"""
raw image processing
"""

import numpy as np
import rawpy
import exifread
import matplotlib.pyplot as plt
import os
from PIL import Image
from io import BytesIO


def extract_meta(file):
  """
  extract meta data from file
  """
  with open(file, 'rb') as f: [print('key: {} value: {}'.format(k, v)) for k, v in exifread.process_file(f).items() if k != 'JPEGThumbnail']


def plot_image(x, name='name', plot_path=None, show_plot=False, close_plot=True):
  """
  plot mfcc extracted features only (no time series)
  """

  # setup figure
  fig = plt.figure()

  # create axis
  ax = plt.axes()

  # plot selected mfcc
  im = ax.imshow(x)

  # axis off
  plt.axis('off'), ax.axis('off')

  # plot the fig
  if plot_path is not None: plt.savefig(plot_path + name + '.png', dpi=100)
  if show_plot: plt.show()
  if close_plot: plt.close()


def thumbnailing(raw, file_name):
  """
  thumbnailing
  """

  try: thumb = raw.extract_thumb()
  except: rawpy.LibRawNoThumbnailError: print("no thumbnail")
  else:

    # thumb format
    if thumb.format in [rawpy.ThumbFormat.JPEG, rawpy.ThumbFormat.BITMAP]:

      if thumb.format is rawpy.ThumbFormat.JPEG:

        thumb_file_name = file_name.split('.')[0] + '_thumb.jpg'
        #with open(thumb_file_name, 'wb') as f: f.write(thumb.data)
        #img = Image.fromarray(thumb.data)
        img = Image.open(BytesIO(thumb.data))
        img.save(thumb_file_name)





if __name__ == '__main__':
  """
  main
  """

  # image path
  img_path = '/world/nest/fotos/Fotografie/ToDo/2021-09-09_Skandinavia/origin/'

  # output path
  plot_path = img_path + 'ignore/process/'

  # create folders
  if not os.path.isdir(plot_path): os.makedirs(plot_path)

  # file format
  file_format = '.CR2'
 
  # file name
  file_name = 'H51A1690' + file_format

  # read file
  with rawpy.imread(img_path + file_name) as raw:

    # thumbnailing
    #thumbnailing(raw, file_name)

    # colors
    print(raw.num_colors)

    # rgb values
    rgb1 = raw.postprocess()
    rgb2 = raw.postprocess(rawpy.Params(use_camera_wb=True))

    # plot images
    #plot_image(rgb1, name='raw_post', plot_path=plot_path, show_plot=False, close_plot=False)
    #plot_image(rgb2, name='raw_post_wb', plot_path=plot_path, show_plot=False, close_plot=False)

  