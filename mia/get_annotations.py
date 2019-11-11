import numpy as np

def get_annotations(file_name):
  """ get_annotations(file_name)
  get hand labeled time instances from text file written by Sonic Visualizer
  with format:
    time | point_name
  
  importing this function:
    from get_annotations import get_annotations


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



# --
# Main function
if __name__ == '__main__':

  # file location and name
  file_path = "./ignore/sounds/"
  file_name = "megalovania.txt"

  # print the phonem labels
  print(get_annotations(file_path + file_name))