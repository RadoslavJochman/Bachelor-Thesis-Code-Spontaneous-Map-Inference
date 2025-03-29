"""
Author: Radoslav Jochman
"""
import argparse
import helper
parser = argparse.ArgumentParser()
parser.add_argument("--data_location", default=None, type=str, help="Path to a directory with data or a text file listing paths to individual data files or path to csv with saved data.")
parser.add_argument("--ref_obj_path", default=None, type=str, help="Path to a object used as the reference for RMSE computation")
parser.add_argument("--result_dir", default=None, type=str, help="Directory where output results will be stored.")
parser.add_argument("--PCs", default=None, type=str, help="Principal Components to use in the analysis (comma-separated).")
parser.add_argument("--result_name", default=None, type=str, help="Name of the graph.")

def main(args: argparse.Namespace):
    paths = helper.extract_pickle_paths(args.data_location)
    PCs = list(map(int,args.PCs.split(",")))
    rmse_dist = helper.calculate_rmse_distr_for_sample(paths,PCs,args.ref_obj_path,"Sample_1")
    rmse_dist.to_csv(f"{args.result_dir}/{args.result_name}")
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)