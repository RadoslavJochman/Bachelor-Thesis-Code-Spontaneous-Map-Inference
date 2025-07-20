"""
Script: compute_rmse_vs_real_monkey.py

Author: Radoslav Jochman

Description:
    This script computes RMSE between spontaneous maps from a
    sample and a real monkey reference recording. The RMSE is calculated after optimally
    rotating the sample's spontaneous maps to align with the reference map.

    The analysis is repeated across all pairwise combinations of specified principal components (PCs).

Usage:
    python compute_rmse_vs_real_monkey.py
        --data_location path/to/sample_data
        --ref_sample_location path/to/monkey_reference.pkl
        --result_dir path/to/output
        --PCs 0,1,2
        --sample_name SampleX

Arguments:
    --data_location : str
        Path to a directory or a text file listing `.pkl` files representing sample recordings.
    --ref_sample_location : str
        Path to the `.pkl` file representing the real monkey reference data.
    --result_dir : str
        Directory where the resulting CSV file will be saved.
    --PCs : str
        Comma-separated list of principal components to be used (e.g., "0,1,2").
    --sample_name : str
        Name of the sample; used for labeling and naming output.

Output:
    A CSV file named `<sample_name>.csv` containing:
        - sample: Name of the sample.
        - bin_size: Time bin size used in preprocessing.
        - PC_pair: The pair of PCs used (e.g., "1,2").
        - TH: Threshold used for preprocessing.
        - RMSE: Root Mean Square Error between aligned and reference maps.
"""
import itertools
from array_analysis import load_object
import pandas as pd
from helper import get_TH, get_bin_size, find_ideal_rotation, rmse_angles
import argparse
import helper

parser = argparse.ArgumentParser()
parser.add_argument("--data_location", default=None, type=str, help="Path to a directory with data or a text file listing paths to individual data files")
parser.add_argument("--ref_sample_location", default=None, type=str, help="Path to a directory with reference data or a text file listing paths to individual reference data files")
parser.add_argument("--result_dir", default=None, type=str, help="Directory where output results will be stored.")
parser.add_argument("--PCs", default=None, type=str, help="Principal Components to use in the analysis (comma-separated).")
parser.add_argument("--sample_name", default=None, type=str, help="Name of the sample.")

def calculate_rmse_distr_sample_and_real_monkey(paths: list[str],PCs: list[int], ref_path: str, sample: str):
    results_list =[]
    paths.sort()
    PCs.sort()
    ref_obj = load_object(ref_path)
    for path in paths:
        TH = get_TH(path)
        bin_size = get_bin_size(path)
        arr_obj = load_object(path)
        for PC1, PC2 in itertools.combinations(PCs,2):
            arr_obj.compute_new_PC(PC1, PC2)
            arr_map = find_ideal_rotation(ref_obj.spontaneous_map, arr_obj.spontaneous_map)
            rmse = rmse_angles(arr_map, ref_obj.spontaneous_map)
            results_list.append([sample, bin_size, f"{PC1+1},{PC2+1}", TH, rmse])
    result = pd.DataFrame(results_list, columns=["sample", "bin_size", "PC_pair", "TH", "RMSE"])

    return result

def main(args: argparse.Namespace):
    paths = helper.extract_paths(args.data_location, ".pkl")
    ref_path = args.ref_sample_location
    PCs = list(map(int,args.PCs.split(",")))
    rmse_dist = calculate_rmse_distr_sample_and_real_monkey(paths,PCs,ref_path,args.sample_name)
    rmse_dist.to_csv(f"{args.result_dir}/{args.sample_name}.csv",index=False)
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


