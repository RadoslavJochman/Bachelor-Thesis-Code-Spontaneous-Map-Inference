"""
Script: filter_spikes.py

Author: Radoslav Jochman

Description:
    This script filters neural spike train data based on specified quality metrics
    such as signal-to-noise ratio (SNR), presence ratio, firing rate, and ISI violation ratio.

    It processes NIX files in the given directory, extracts spike train annotations,
    applies filtering criteria across all datasets, and outputs the channels that
    meet all criteria as a CSV file.

Usage:
    python filter_spikes.py
        --spikes_dir path/to/spikes_data
        --snr 5.0
        --presence_ratio 0.9
        --max_firing_rate 100
        --min_firing_rate 0.5
        --isi_violations_ratio 0.1
        --result_path filtered_channels.csv

Arguments:
    --spikes_dir : str
        Directory or text file listing NIX spike data files.
    --snr : float
        Minimum signal-to-noise ratio to keep a channel.
    --presence_ratio : float
        Minimum presence ratio to keep a channel.
    --max_firing_rate : float, optional (default: infinity)
        Maximum firing rate to keep a channel.
    --min_firing_rate : float, optional (default: 0)
        Minimum firing rate to keep a channel.
    --isi_violations_ratio : float
        Maximum inter-spike interval violation ratio to keep a channel.
    --result_path : str
        File path to save the filtered channel IDs as CSV
"""
import argparse
import neo
import pandas as pd

from helper import extract_paths

parser = argparse.ArgumentParser()
parser.add_argument("--spikes_dir", default=None, type=str, help="Path to the directory containing input data or a text file listing the data file paths.")
parser.add_argument("--snr", default=None, type=float, help="Minimal SNR used for filtering")
parser.add_argument("--presence_ratio", default=None, type=float, help="Minimal presence ratio used for filtering")
parser.add_argument("--max_firing_rate", default=float("inf"), type=float, help="Maximal firing rate used for filtering")
parser.add_argument("--min_firing_rate", default=0, type=float, help="Minimal firing rate used for filtering")
parser.add_argument("--isi_violations_ratio", default=None, type=float, help="Maximal isi violations ratio used for filtering")
parser.add_argument("--result_path", default=None, type=str, help="Path where CSV will be saved.")


def main(args):
    paths = extract_paths(args.spikes_dir, "nix")
    channel_id = []
    for path in paths:
        block = neo.NixIO(path).read_block()
        annotations_list = [st.annotations for st in block.segments[0].spiketrains]
        df = pd.DataFrame(annotations_list)
        filtered = df[
            (df["snr"] >= args.snr) &
            (df["presence_ratio"] >= args.presence_ratio) &
            (df["firing_rate"] <= args.max_firing_rate) &
            (df["firing_rate"] >= args.min_firing_rate) &
            (df["isi_violations_ratio"] <= args.isi_violations_ratio)
            ]
        if len(channel_id)==0:
            channel_id = filtered["channel_ids"].unique()
        else:
            channel_id = [channel for channel in channel_id if channel in filtered["channel_ids"]]
    result = pd.DataFrame(channel_id, columns=["channel_ids"])
    result.to_csv(args.result_path, index=False)



if __name__ == '__main__':
    main_args = parser.parse_args()
    main(main_args)