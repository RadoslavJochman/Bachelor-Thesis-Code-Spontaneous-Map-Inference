"""
Script: main_generate_rmse_dist_2samples_param.py

Author: Radoslav Jochman

Description:
    This script computes the RMSE (Root Mean Square Error) distribution between two samples
     based on specified Principal Components (PCs).


Usage:
    python main_generate_rmse_dist_2samples_param.py
        --data_location path/to/sample_data
        --ref_sample_location path/to/reference_data
        --result_dir path/to/output
        --PCs 1,2
        --sample_name SampleA

Arguments:
    --data_location : str
        Directory or text file containing paths to the .pkl files representing the sample.
    --ref_sample_location : str
        Directory or text file containing paths to the .pkl files of reference samples.
    --result_dir : str
        Path to directory where the result CSV will be saved.
    --PCs : str
        Comma-separated list of principal component indices (e.g., "1,2").
    --sample_name : str
        Name identifier for the sample being analyzed; used as the CSV file name.
"""
import argparse
import helper
parser = argparse.ArgumentParser()
parser.add_argument("--data_location", default=None, type=str, help="Path to a directory with data or a text file listing paths to individual data files")
parser.add_argument("--ref_sample_location", default=None, type=str, help="Path to a directory with reference data or a text file listing paths to individual reference data files")
parser.add_argument("--result_dir", default=None, type=str, help="Directory where output results will be stored.")
parser.add_argument("--PCs", default=None, type=str, help="Principal Components to use in the analysis (comma-separated).")
parser.add_argument("--sample_name", default=None, type=str, help="Name of the sample.")

def main(args: argparse.Namespace):
    paths = helper.extract_paths(args.data_location, ".pkl")
    ref_paths = helper.extract_paths(args.ref_sample_location, ".pkl")
    PCs = list(map(int,args.PCs.split(",")))
    rmse_dist = helper.calculate_rmse_distr_sample_and_ref_sample(paths,PCs,ref_paths,args.sample_name)
    rmse_dist.to_csv(f"{args.result_dir}/{args.sample_name}.csv",index=False)
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)