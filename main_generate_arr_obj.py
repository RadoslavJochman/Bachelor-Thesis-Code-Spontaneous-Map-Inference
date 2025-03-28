"""
Author: Radoslav Jochman
"""
import pandas as pd
import yaml
import argparse
from analysis import compute_spontaneous_map, compute_frames
from array_analysis import ArrayAnalysis, METHODS
from helper import load_human_segments



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

    with open(args.analysis_params_dir) as f:
        analysis_params = yaml.safe_load(f)

    layout = pd.read_csv(params["layout_path"])
    arr_obj = ArrayAnalysis("4",METHODS.nLFP,segments,layout,analysis_params)

    compute_spontaneous_map(arr_obj)
    PCs = args.PCs.split(",")
    arr_obj.compute_new_PC(PCs[0],PCs[1])
    arr_obj.save_lightweight(f'{args.result_dir}/{args.result_name}')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)