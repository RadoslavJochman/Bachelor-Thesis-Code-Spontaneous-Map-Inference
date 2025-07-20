"""
Script: main_compute_new_PC.py

Author: Radoslav Jochman

Description:
    This script processes neural data segments to compute and save spontaneous activity
    maps projected onto specified principal component dimensions.

    It loads data and analysis parameters, converts LFP segments to nLFP format,
    initializes an ArrayAnalysis object, computes spontaneous maps, projects the data
    onto selected PCA components, and saves the result as a pickled object.

Usage:
    python main_compute_new_PC.py
        --analysis_params_dir path/to/analysis_params.yaml
        --params_dir path/to/data_params.yaml
        --data_dir path/to/data_or_list.txt
        --PCs "0,1"
        --result_dir path/to/save/results
        --result_name output_filename.pkl

Arguments:
    --analysis_params_dir : str
        Path to YAML file containing parameters for the analysis.
    --params_dir : str
        Path to YAML file containing parameters for data loading and preprocessing.
    --data_dir : str
        Directory or file listing neural data segments to process.
    --PCs : str
        Comma-separated indices of principal components to use (e.g., "0,1").
    --result_dir : str
        Directory where the output pickled object will be saved.
    --result_name : str
        Filename for the output pickle.
"""
import pandas as pd
import yaml
import argparse
from analysis import compute_spontaneous_map, compute_frames
from array_analysis import ArrayAnalysis, METHODS
from helper import load_human_segments, process_LFP_to_nLFP



parser = argparse.ArgumentParser()
parser.add_argument("--analysis_params_dir", default=None,help="YML File containing parameters for analysis")
parser.add_argument("--params_dir", default=None, help="YML File containing parameter for data loading and processing")
parser.add_argument("--data_dir", default=None, help="Data directory or text file with the paths to data")
parser.add_argument("--PCs", default=None, type=str, help="Principal Components to use in the analysis (comma-separated).")
parser.add_argument("--result_dir", default=None, help="Directory where results will be stored")
parser.add_argument("--result_name", default=None, help="Name of the pickled object.")

def main(args: argparse.Namespace):
    with open(args.params_dir) as f:
        params = yaml.safe_load(f)

    segments = load_human_segments(args.data_dir, params)
    segments = process_LFP_to_nLFP(segments, params)

    with open(args.analysis_params_dir) as f:
        analysis_params = yaml.safe_load(f)

    layout = pd.read_csv(params["layout_path"])
    arr_obj = ArrayAnalysis("4",METHODS.nLFP,segments,layout,analysis_params)

    compute_spontaneous_map(arr_obj)
    PC1, PC2 = map(int,args.PCs.split(","))
    arr_obj.compute_new_PC(PC1, PC2)
    arr_obj.save_lightweight(f'{args.result_dir}/{args.result_name}')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)