"""
Author: Radoslav Jochman
"""
import argparse
import pandas as pd
from array_analysis import ArrayAnalysis, METHODS
from analysis import compute_spontaneous_map
from helper import load_human_segments, split_segment, process_LFP_to_nLFP
import yaml
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--analysis_params_path", default=None, type=str, help="Path to the YAML file that defines the parameters for the analysis process.")
parser.add_argument("--params_path", default=None, type=str, help="Path to the YAML file containing configuration parameters for data loading and preprocessing.")
parser.add_argument("--data_location", default=None, type=str, help="Path to a directory with data or a text file listing paths to individual data files.")
parser.add_argument("--result_dir", default=None, type=str, help="Directory where output results will be stored.")
parser.add_argument("--segment_size", default=None, type=int, help="Duration in seconds for each data segment used during the analysis.")
parser.add_argument("--good_channels",type=str, default=None, help="Path to CSV file with channel indices used as good channels.")

def main(args: argparse.Namespace):
    with open(args.analysis_params_path) as f:
        analysis_params = yaml.safe_load(f)
    with open(args.params_path) as f:
        params = yaml.safe_load(f)
    layout = pd.read_csv(params["layout_path"])
    n_channels = np.sum(~np.isnan(layout["chn"]))
    if args.good_channels!=None:
        good_channels = pd.read_csv(args.good_channels)["channel_ids"]
        bad_channels = [channel for channel in range(n_channels) if channel not in good_channels]
    segments = load_human_segments(args.data_location,params)
    for seg in segments:
        for i, split_seg in enumerate(split_segment(seg, args.segment_size)):
            split_seg = process_LFP_to_nLFP([split_seg],params)
            arr_obj = ArrayAnalysis("4",METHODS.nLFP,split_seg,layout,analysis_params)
            arr_obj.deleted_channels=bad_channels
            compute_spontaneous_map(arr_obj)
            arr_obj.save_lightweight(f"{args.result_dir}/TH_fac_{params['threshold_factor']}_bin_size_{analysis_params['event_binsize']}_start_{i*args.segment_size//60}.pkl")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)