import glob
import sys
import os

dataset_folder = sys.argv[1]

folders = []
log_files = []

for coords in glob.glob(f"{dataset_folder}/*/*.unipartite.coordinates"):
    folder = os.path.dirname(coords)
    folders.append(f"'{coords}'")

folders = " ".join(folders)

command = 'parallel --link --joblog my.log python scripts/model_maximum_accuracy.py -c {1} -l {2}' + f' ::: {folders}'
os.system(command)
