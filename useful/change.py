"""
change detection of files in folders for backup
"""

import os
import sys

import numpy as np

from glob import glob
from pathlib import Path



def check_sub_dirs(sub_p1, sub_p2, rel_p1, rel_p2):
	"""
	check if sub dirs contain same folders
	"""

	# prints
	print("\nlen-is_okay: [{}]".format(len(sub_p1) == len(sub_p2)))
	print("sub_p1: ", [str(f.relative_to(rel_p1)) for f in sub_p1])
	print("sub_p2: ", [str(f.relative_to(rel_p2)) for f in sub_p2])

	# comparison
	comp = [a.relative_to(p1) == b.relative_to(p2) for a, b in zip(sub_p1, sub_p2)]
	print(comp)

	# exit
	if not all(comp) == True or len(sub_p1) != len(sub_p2):

		# get wrong indices
		wrong_idx = [i for i, x in enumerate(comp) if not x]

		# prints
		print("\n wrong names at: ", np.array(sub_p1)[wrong_idx])

		print("\n***sub dirs failed")
		sys.exit()



def check_sums(s1, s2, names):
	"""
	compare sums
	"""

	# prints
	print("s1: ", s1)
	print("s2: ", s2)

	comp = [a == b for a, b in zip(s1, s2)]
	print(comp)

	# exit
	if not all(comp) == True or len(s1) != len(s2):

		# get wrong indices
		wrong_idx = [i for i, x in enumerate(comp) if not x]

		# prints
		print("\n wrong sums at: ", np.array(names)[wrong_idx])
		print("\n***checksum failed")
		sys.exit()


if __name__ == '__main__':

	# argument check
	if len(sys.argv) != 2 + 1:

		# too much arguments
		print("--Usage: python change.py [root_dir1] [root_dir2]")

		# exit
		sys.exit()


	# --
	# change detection

	p1, p2 = Path(sys.argv[1]), Path(sys.argv[2])

	# print message
	print("\n--Change detection:")
	print("path1: [{}], path2: [{}]".format(p1, p2))

	# sub paths
	sub_p1 = [f for f in sorted(p1.iterdir()) if f.is_dir()]
	sub_p2 = [f for f in sorted(p2.iterdir()) if f.is_dir()]

	# check subs
	check_sub_dirs(sub_p1, sub_p2, p1, p2)

	# go through subs
	for s_p1, s_p2 in zip(sub_p1, sub_p2):

		print("\nsubs: {}, {}".format(str(s_p1), str(s_p2)))

		# sub paths
		subsub_p1 = [f for f in sorted(s_p1.iterdir()) if f.is_dir()]
		subsub_p2 = [f for f in sorted(s_p2.iterdir()) if f.is_dir()]

		# check subs
		check_sub_dirs(subsub_p1, subsub_p2, p1, p2)

		# check file space
		#for ss_p1, ss_p2 in zip(subsub_p1, subsub_p2):

		s1 = [sum(f.stat().st_size for f in ss_p1.glob('**/*') if f.is_file()) for ss_p1 in subsub_p1]
		s2 = [sum(f.stat().st_size for f in ss_p2.glob('**/*') if f.is_file()) for ss_p2 in subsub_p2]

		# check sums
		check_sums(s1, s2, subsub_p1)


	print("\n---Everything okay :)")



