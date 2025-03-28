import pandas as pd

from array_analysis import ArrayAnalysis, METHODS
from blackrock_utilities.brpylib             import NevFile, NsxFile, brpylib_ver
import yaml
from helper import load_human_segments
from analysis import compute_spontaneous_map
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--analysis_params_dir", default=None, type=str, help="Path to the YAML file containing analysis-specific parameters")
parser.add_argument("--params_dir", default=None, type=str, help="Path to the YAML file containing data loading and preprocessing parameters")
parser.add_argument("--data_dir", default=None, type=str, help="Path to the directory containing input data or a text file listing the data file paths.")
parser.add_argument("--result_dir", default=None, type=str,help="Directory where the output files (e.g., pickled objects, analysis results) will be saved.")
parser.add_argument("--start_size", type=float, default=0.15, help="Starting value of the bin size range to explore (in seconds).")
parser.add_argument("--end_size", type=float, default=5.0, help="Ending value of the bin size range to explore (in seconds).")
parser.add_argument("--num_steps", type=int, default=200, help="Number of steps to divide the bin size range.")

def main(args):
    bin_sizes = np.round(np.linspace(args.start_size, args.end_size, args.num_steps), 2)
    with open(args.params_dir) as f:
        params = yaml.safe_load(f)

    th_factor = params["threshold_factor"]
    with open(args.analysis_params_dir) as f:
        params_analysis = yaml.safe_load(f)

    layout = pd.read_csv(params['layout_path'])

    #go through all analysed bin sizes and create ArrayAnalysis object
    for bin_size in bin_sizes:
        params_analysis["event_binsize"] = bin_size

        segments = load_human_segments(args.data_dir, params['layout_path'],
                                  subtract_mean=params['subtract_mean'],
                                  threshold_factor=params['threshold_factor'],
                                  filter=params['filter'], z_score=params['z-score'])

        arr_obj = ArrayAnalysis("4",METHODS.nLFP,segments,layout,params_analysis)
        compute_spontaneous_map(arr_obj)
        arr_obj.save_lightweight(f'{args.result_dir}/TH_fac_{th_factor}_bin_size_{bin_size}.pkl')
        del arr_obj

if __name__ == '__main__':
    main_args = parser.parse_args()
    main(main_args)