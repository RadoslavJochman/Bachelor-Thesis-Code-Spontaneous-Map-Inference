"""
Script: main_params_analysis.py

Author: Radoslav Jochman

Description:
    This script executes a full processing pipeline for human intracranial LFP recordings. It converts
    raw LFP signals into (nLFPs), splits the data into time segments, and computes
    spontaneous activity maps across a range of bin sizes. Results are saved as lightweight pickled
    `ArrayAnalysis` objects for further inspection or visualization.

    Configuration is done using YAML files that define both data preprocessing and analysis-specific
    parameters.

Usage:
    python main_params_analysis.py
        --analysis_params_dir path/to/analysis_params.yaml
        --params_dir path/to/preprocessing_params.yaml
        --data_dir path/to/data_or_list.txt
        --result_dir path/to/output
        --start_size 0.15
        --end_size 5.0
        --num_steps 200
        --segment_size 300
        --good_channels path/to/good_channels.csv

Arguments:
    --analysis_params_dir : str
        Path to a YAML file containing analysis-specific parameters (e.g., bin size, thresholding strategy).
    --params_dir : str
        Path to a YAML file containing preprocessing parameters (e.g., filtering, normalization).
    --data_dir : str
        Directory containing input LFP data files or a text file listing file paths.
    --result_dir : str
        Directory where output `.pkl` files with analysis results will be saved.
    --start_size / --end_size / --num_steps : float/int
        Defines the range and resolution of bin sizes (in seconds) to scan for spontaneous activity.
    --segment_size : int
        Duration (in seconds) of data segments to split recordings into for analysis.
    --good_channels : str
        CSV file with a list of valid channel indices. Remaining channels will be excluded from analysis.

Output:
    - Directory structure: `<result_dir>/sub_sample_<index>/TH_fac_<factor>_bin_size_<bin_size>.pkl`
    - Each file contains an `ArrayAnalysis` object with metadata and results.
"""
import os

import pandas as pd

import helper
from array_analysis import ArrayAnalysis, METHODS
import yaml
from helper import load_human_segments, process_LFP_to_nLFP, split_segment
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
parser.add_argument("--segment_size", default=None, type=int, help="Duration in seconds for each data segment used during the analysis.")

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
    for seg in segments:
        for i, split_seg in enumerate(split_segment(seg, args.segment_size)):
            split_seg = process_LFP_to_nLFP([split_seg],params)
            for bin_size in bin_sizes:
                params_analysis["event_binsize"] = bin_size
                arr_obj = ArrayAnalysis("4",METHODS.nLFP,split_seg,layout,params_analysis)
                arr_obj.deleted_channels=bad_channels
                compute_spontaneous_map(arr_obj)
                result_dir= f'{args.result_dir}/sub_sample_{i}'
                os.makedirs(result_dir, exist_ok=True)
                arr_obj.save_lightweight(f'{result_dir}/TH_fac_{th_factor}_bin_size_{bin_size}.pkl')
                del arr_obj

if __name__ == '__main__':
    main_args = parser.parse_args()
    main(main_args)