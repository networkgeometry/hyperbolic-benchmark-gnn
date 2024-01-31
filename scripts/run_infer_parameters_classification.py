import glob
import sys
import os

dataset_folder = sys.argv[1]
folders = []

for coords in glob.glob(f"{dataset_folder}/*/*.unipartite.coordinates"):
    folder = os.path.dirname(coords)
    folders.append(f"'{coords}'")

folders = " ".join(folders)
command = 'parallel --link --joblog my.log python scripts/infer_parameters_classification.py -c {1}' + f' ::: {folders}'
os.system(command)
