import numpy as np

# --
# get phonem labels and time stamps out of file
def get_phonemlabels(file):
  """ get_phonemlabels(filename)
  get phonem label and time stamps of the format:
    start time, end time and label
  
  importing this package:
    from get_phonemlabels import get_phonemlabels

  call this in matplotlib:
    # plot phonems in normed plot
    for ph in phonems:

      # start time limit
      if float(ph[0]) < t_roi[0]: 
        continue; 

      # stop time limit
      if float(ph[0]) > t_roi[1]: 
        break; 

      # draw vertical lines
      plt.axvline(x=float(ph[0]), dashes=(1, 1), color='k')

      # write phone label
      plt.text(x=float(ph[0]), y=0.9, s=ph[2], color='k', fontweight='semibold')
  """

  # open file
  file = open(file)

  # setup list
  phonem_list = np.empty((0, 3), dtype="<U20")

  # run through all lines
  line_nr = 1;
  for line in file:

    # init phonem data sample
    phonem_data = np.empty(3, dtype='<U20')

    # add start time
    phonem_data[0] = line.rstrip()

    # add end time and phonem label
    for idx in range(1, 3):
      phonem_data[idx] = file.readline().rstrip()
    
    # remove '
    phonem_data[2] = phonem_data[2].split("'")[1]

    # stack the phonems together
    phonem_list = np.vstack((phonem_list, phonem_data))

  # close file
  file.close()

  return phonem_list



# --
# Main function
if __name__ == '__main__':

  # file location and name
  file_path = "./ignore/"
  file_name = "A0101_Phonems.txt"

  # print the phonem labels
  print(get_phonemlabels(file_path + file_name))