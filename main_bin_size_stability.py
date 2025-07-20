"""
Script: main_generate_spontaneous_map.py

Author: Radoslav Jochman

Description:
    This script performs analysis on neural data to generate spontaneous activity maps
    for a range of bin sizes. It processes human neural recording segments (LFP),
    converts them to nLFP signals, and uses the ArrayAnalysis class to compute spontaneous maps.

    The results are saved as lightweight pickled objects for further use.

Usage:
    python main_generate_spontaneous_map.py
        --analysis_params_dir path/to/analysis_params.yaml
        --params_dir path/to/data_params.yaml
        --data_dir path/to/data_or_list.txt
        --result_dir path/to/save/results
        [--start_size float] [--end_size float] [--num_steps int]
        [--good_channels path/to/good_channels.csv]

Arguments:
    --analysis_params_dir : str
        Path to YAML file with analysis-specific parameters (e.g., event bin sizes).
    --params_dir : str
        Path to YAML file with data loading and preprocessing parameters.
    --data_dir : str
        Directory or text file listing input neural data segments.
    --result_dir : str
        Directory to save output pickled analysis results.
    --start_size : float, optional
        Starting bin size for event detection (default 0.15 seconds).
    --end_size : float, optional
        Ending bin size for event detection (default 5.0 seconds).
    --num_steps : int, optional
        Number of bin sizes to evaluate between start and end (default 200).
    --good_channels : str, optional
        CSV file listing channels to use as "good"; others will be marked deleted.
"""
import pandas as pd

import helper
from array_analysis import ArrayAnalysis, METHODS
import yaml
from helper import load_human_segments, process_LFP_to_nLFP
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
parser.add_argument("--good_channels",type=str, default=None, help="Path to CSV file with channel indices used as good channels.")

def main(args):
    bin_sizes = np.round(np.linspace(args.start_size, args.end_size, args.num_steps), 2)
    with open(args.params_dir) as f:
        params = yaml.safe_load(f)

    th_factor = params["threshold_factor"]
    with open(args.analysis_params_dir) as f:
        params_analysis = yaml.safe_load(f)

    layout = pd.read_csv(params['layout_path'])
    n_channels = np.sum(~np.isnan(layout["chn"]))
    bad_channels = []
    if args.good_channels!=None:
        good_channels = pd.read_csv(args.good_channels)["channel_ids"]
        bad_channels = [channel for channel in range(n_channels) if channel not in good_channels]
    #go through all analysed bin sizes and create ArrayAnalysis object
    segments = load_human_segments(args.data_dir, params)
    segments = process_LFP_to_nLFP(segments,params)
    for bin_size in bin_sizes:
        params_analysis["event_binsize"] = bin_size

        arr_obj = ArrayAnalysis("4",METHODS.nLFP,segments,layout,params_analysis)
        arr_obj.deleted_channels=bad_channels
        compute_spontaneous_map(arr_obj)
        arr_obj.save_lightweight(f'{args.result_dir}/TH_fac_{th_factor}_bin_size_{bin_size}.pkl')
        del arr_obj

if __name__ == '__main__':
    main_args = parser.parse_args()
    main(main_args)