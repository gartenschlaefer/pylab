import os
import sys
from glob import glob

def main(file_path, rename_name='file'):
  """
  main function for file renamer
  """

  # run through each file
  #for i, filename in enumerate(glob(file_path + '/' + '*.*')):
  for i, filename in enumerate(sorted(glob(file_path + '/' + '*.*'), key=os.path.getmtime)):

    # save the own file
    if filename == 'file_renamer.py':
      continue

    # get file ext
    file_ext = filename.split('.')[-1]

    # new name
    new_name = file_path + '/' + rename_name + '-' + str(i) + '.' + file_ext

    # print message
    print("rename file: [{}] to: [{}]".format(filename, new_name))

    os.rename(filename, new_name)



if __name__ == '__main__':

  # check arg length
  if len(sys.argv) != 2 + 1:

    # too much arguments
    print("--Usage: python file_renamer.py [path_to_files] [Rename_name]")

  else:

    # print message
    print("path: [{}], rename_name: [{}]".format(sys.argv[1], sys.argv[2]))
    
    # run main
    main(sys.argv[1], sys.argv[2])