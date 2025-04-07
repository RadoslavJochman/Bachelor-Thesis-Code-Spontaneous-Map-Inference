"""
Author: Radoslav Jochman
"""
import argparse
import itertools

import helper
import pandas as pd
from array_analysis import load_object
parser = argparse.ArgumentParser()
parser.add_argument("--data_location", default=None, type=str, help="Path to a directory with data or a text file listing paths to individual data files")
parser.add_argument("--result_dir", default=None, type=str, help="Directory where output results will be stored.")
parser.add_argument("--sample_name", default=None, type=str, help="Name of the sample.")

def main(args: argparse.Namespace):
    paths = helper.extract_paths(args.data_location, ".pkl")
    paths.sort()
    results_list = []
    for sample_1, sample_2 in itertools.combinations(paths,2):
        obj_1 = load_object(sample_1)
        obj_2 = load_object(sample_2)
        time_1 = sample_1.split("/")[-1][-5]
        time_2 = sample_2.split("/")[-1][-5]
        spont_1 = obj_1.spontaneous_map
        spont_2 = obj_2.spontaneous_map
        spont_1 = helper.find_ideal_rotation(spont_1,spont_2)
        rmse = helper.rmse_angles(spont_1,spont_2)
        results_list.append([args.sample_name,f"{abs(int(time_1) - int(time_2))}",rmse])
    rmse_dist = pd.DataFrame(results_list, columns=["sample", "time_diff" ,"RMSE"])
    rmse_dist.to_csv(f"{args.result_dir}/{args.sample_name}.csv")
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)