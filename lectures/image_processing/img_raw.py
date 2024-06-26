"""
raw image processing
refs:
https://www.numbercrunch.de/blog/2020/12/from-numbers-to-images-raw-image-processing-with-python/
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
  with open(file, 'rb') as f: [print('{}: {}'.format(k, v)) for k, v in exifread.process_file(f).items() if k != 'JPEGThumbnail']


def extract_raw_data(raw):
  """
  print infos
  """
  print('\n--\ncamera infos:\n\nraw type: {}\nnum colors: {}\ncolor description: {}\nraw pattern:\n{}\nblack levels: {}\nwhite level: {}\ncolor matrix:\n{}\nxyz to rgb matrix:\n{}\ncam white balance: {}\ndayligt white balance {}\n--\n'.format(
    raw.raw_type, raw.num_colors, raw.color_desc, raw.raw_pattern, raw.black_level_per_channel, raw.white_level, raw.color_matrix, raw.rgb_xyz_matrix, raw.camera_whitebalance, raw.daylight_whitebalance))


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


def save_jpeg(x, name, plot_path):
  """
  save as jpeg
  """

  # read as array
  img = Image.fromarray(x)

  # save file
  img.save(plot_path + name)


def simple_image_processing(raw):
  """
  simple image processing pipeline
  """

  # get raw data
  img = np.array(raw.raw_image, dtype=np.double)
  print("img: ", img.shape)

  # create black matrix of image size
  black = np.tile(np.array(raw.black_level_per_channel, dtype=np.double).reshape(2, 2), (img.shape[0] // 2, img.shape[1] // 2))

  # substract black and normalize
  img = (img - black) / (raw.white_level - black)

  # colors
  colors = np.frombuffer(raw.color_desc, dtype=np.byte)
  pattern = np.array(raw.raw_pattern)

  # index
  indices = [np.where(colors[pattern] == colors[i]) for i in range(4)]

  # white balance init
  wb = np.zeros((2, 2), dtype=np.double)

  # normalize by green color channel
  for i in range(raw.num_colors): wb[indices[i]] = raw.camera_whitebalance[i] / raw.camera_whitebalance[1]

  # scale up
  wb = np.tile(wb, (img.shape[0] // 2, img.shape[1] // 2))

  # apply white balance
  img_wb = np.clip(img * wb, 0, 1)


  # demosaic image for downsampling (bad method, but reduces space)
  img_demosaic = np.empty((img_wb.shape[0] // 2, img_wb.shape[1] // 2, raw.num_colors))

  # appyl demoasaic downsampling
  for i in range(raw.num_colors): img_demosaic[:, :, i] = img_wb[indices[i][0][0]::2, indices[i][1][0]::2] if i != 1 else (img_wb[indices[i][0][0]::2, indices[i][1][0]::2] + img_wb[indices[i][0][1]::2, indices[i][1][1]::2]) / 2
  print(img_demosaic.shape)

  # sRGB
  A = np.array(raw.rgb_xyz_matrix[0:raw.num_colors, :], dtype=np.double)
  B = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]], dtype=np.double)
  C = A @ B

  # normalize
  C /= np.tile(np.sum(C, 1), (3, 1)).T

  # invert sRGB matrix
  C_inv = np.linalg.inv(C)
  print("cam to srgb inv: ", C_inv)

  # camera to sRGB
  img_srgb = np.einsum('ij,...j', C_inv, img_demosaic)
  print(img_srgb.shape)

  # gamma correction
  img_srgb_gamma = img_srgb.copy()

  # gamma curve
  img_srgb_gamma[img_srgb < 0.0031308] *= 323 / 25 
  img_srgb_gamma[np.logical_not(img_srgb_gamma < 0.0031308)] = 211 / 200 * img_srgb_gamma[np.logical_not(img_srgb_gamma < 0.0031308)] ** (5 / 12) - 11 / 200
  
  # clipping
  img_srgb_gamma = np.clip(img_srgb_gamma, 0, 1)
  img_srgb = np.clip(img_srgb, 0, 1)

  # save images
  [save_jpeg(np.array(x * 255, dtype=np.uint8), '{}_{}_{}{}'.format(file_name, i, n, o_format), plot_path) for i, (x, n) in enumerate(zip([img, img_wb, img_demosaic, img_srgb, img_srgb_gamma], ['raw', 'raw_wb', 'demosaic', 'srgb', 'srgb_gamma']))]



def rawpy_processing(raw):
  """
  rawpy processing pipline
  """

  # rgb values
  img_rawpy = raw.postprocess()
  img_rawpy_wb = raw.postprocess(rawpy.Params(use_camera_wb=True))

  # save rawpy processing
  [save_jpeg(x, '{}_{}_{}{}'.format(file_name, i + 10, n, o_format), plot_path) for i, (x, n) in enumerate(zip([img_rawpy, img_rawpy_wb], ['rawpy', 'rawpy_wb']))]



if __name__ == '__main__':
  """
  main
  """

  # image path
  img_path = '/world/nest/fotos/Fotografie/ToDo/2021-09-09_Skandinavia/origin/'

  # output path
  #plot_path = img_path + 'ignore/'
  plot_path = './ignore/'

  # create folders
  if not os.path.isdir(plot_path): os.makedirs(plot_path)

  # file format
  i_format = '.CR2'
  o_format = '.jpg'
 
  # file name
  file_name = 'H51A1690'

  print("file: ", img_path + file_name)

  # meta
  #extract_meta(img_path + file_name)

  # read file
  with rawpy.imread(img_path + file_name + i_format) as raw:

    # raw data
    extract_raw_data(raw)


    # thumbnailing
    #thumbnailing(raw, file_name)

    # simple image processing
    #simple_image_processing(raw)

    # rawpy processing
    rawpy_processing(raw)

