"""
Script: main_generate_rmse_dist_param.py

Author: Radoslav Jochman

Description:
    This script computes the RMSE (Root Mean Square Error) between spontaneous maps
    from a given sample and a specified reference object.

    Each spontaneous map from the sample is aligned with the reference map using optimal rotation
    before RMSE calculation. The comparison is done over specified principal components (PCs).

Usage:
    python main_generate_rmse_dist_param.py
        --data_location path/to/sample_data
        --ref_obj_path path/to/reference.pkl
        --result_dir path/to/output
        --PCs 0,1
        --sample_name SampleA

Arguments:
    --data_location : str
        Directory, text file listing `.pkl` data files, or CSV containing the data.
    --ref_obj_path : str
        Path to the reference `.pkl` object used for RMSE comparison.
    --result_dir : str
        Directory where the resulting CSV file will be stored.
    --PCs : str
        Comma-separated principal components used for map comparison (e.g., "0,1").
    --sample_name : str
        Name of the sample being processed; used in the output filename.

Output:
    A CSV file named `<sample_name>.csv` containing:
        - sample: name of the sample
        - timepoint: (optional, depends on helper)
        - RMSE: RMSE between aligned spontaneous map and reference
"""

import argparse
import helper
parser = argparse.ArgumentParser()
parser.add_argument("--data_location", default=None, type=str, help="Path to a directory with data or a text file listing paths to individual data files or path to csv with saved data.")
parser.add_argument("--ref_obj_path", default=None, type=str, help="Path to a object used as the reference for RMSE computation")
parser.add_argument("--result_dir", default=None, type=str, help="Directory where output results will be stored.")
parser.add_argument("--PCs", default=None, type=str, help="Principal Components to use in the analysis (comma-separated).")
parser.add_argument("--sample_name", default=None, type=str, help="Name of the sample.")

def main(args: argparse.Namespace):
    paths = helper.extract_paths(args.data_location, ".pkl")
    PCs = list(map(int,args.PCs.split(",")))
    rmse_dist = helper.calculate_rmse_distr_for_sample(paths,PCs,args.ref_obj_path,args.sample_name)
    rmse_dist.to_csv(f"{args.result_dir}/{args.sample_name}.csv",index=False)
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)