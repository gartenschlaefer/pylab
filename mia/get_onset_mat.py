import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile, loadmat



def get_onset_mat(file_name, var_name):
  """
  reads a .mat file with midi notes and gives back
  onsets and the corresponding midi notes
  """
  # read mat files
  mat = loadmat(file_name)

  # get midi notes with funcdamental frequencies
  m = np.round(mat[var_name])

  # gradients of notes
  onsets = np.diff(m)

  # set onsets to one
  onsets[np.abs(onsets) > 0] = 1

  return (onsets, m)




if __name__ == '__main__':

  # file name to mat file
  file_name = '01-AchGottundHerr-GTF0s.mat'

  # var name
  var_name = 'GTF0s'

  # get onsets
  onsets, m = get_onset_mat(file_name, var_name)

  # time scale
  T = 0.01
  t = np.arange(0.023, 0.023 + T * m.shape[1], T)

  # plot
  plt.figure(3, figsize=(8, 4))

  # midi notes
  plt.plot(t, m.T)

  # onsets
  for i in np.arange(0, len(onsets)):
    on = onsets[i, :] * m[i, :-1]
    plt.scatter(t[:-1][on > 0], on[on > 0], label='voice' + str(i+1))

  # titles
  plt.title(file_name)
  plt.ylabel('midi note')
  plt.xlabel('time [s]')

  plt.grid()
  plt.legend()

  plt.savefig('onsets_' + file_name.split('.')[0] + '.png', dpi=150)
  plt.show()



  

