import data_processing as dp
from os import listdir
from os.path import isfile, join

shots_lists_dir = '../data/shot_lists/'
paths = [f for f in listdir(shots_lists_dir) if isfile(join(mypath, f))]

for path in paths:
	dp.clean_shots_list(path)