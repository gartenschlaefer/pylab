# --
# annotations

import numpy as np
import matplotlib.pyplot as plt

def get_annotations(file_name):
  """ 
  get_annotations(file_name)
  get hand labeled time instances from text file written by Sonic Visualizer
  with format: time | point_name
  """

  # open file
  file = open(file_name)

  # setup list
  anno_list = np.empty((0, 1), dtype="float")

  # run through all lines
  for line in file:

    # append time instances
    anno_list = np.append(anno_list, float(line.split()[0]))

  # close file
  file.close()

  return anno_list


def get_annotations_text(file_name):
  """ 
  get_annotations(file_name)
  get hand labeled time instances from text file written by Sonic Visualizer
  with format: time | point_name
  """

  # open file
  file = open(file_name)

  # setup list
  anno_list = np.empty((0, 2))

  # run through all lines
  for line in file:

    split_line = line.split()

    # append time instances
    anno_list = np.vstack((anno_list, (split_line[0], split_line[1])))

  # close file
  file.close()

  return anno_list


def plot_add_anno(file_name, text_height=1):
  """
  adds annotation to plot
  """

  text_bias = 0
  # annotations
  for i, a in enumerate(get_annotations_text(file_name)):

    # draw vertical lines
    plt.axvline(x=float(a[0]), dashes=(1, 1), color='k')

    # add text
    plt.text(x=float(a[0]), y=text_height + text_bias, s=a[1], color='k', fontweight='semibold')

    # text bias
    if text_bias >=1:
      text_bias = -1

    text_bias += 1




# --
# Main function
if __name__ == '__main__':

  # file location and name
  file_path = "./ignore/sounds/"
  file_name = "megalovania.txt"

  # print the phonem labels
  print(get_annotations(file_path + file_name))