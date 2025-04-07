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