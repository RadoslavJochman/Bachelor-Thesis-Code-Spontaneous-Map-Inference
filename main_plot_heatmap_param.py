"""
Author: Radoslav Jochman
"""
import plotting
import argparse
import helper
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("--data_location", default=None, type=str, help="Path to a directory with data or a text file listing paths to individual data files or path to csv with saved data.")
parser.add_argument("--ref_obj_path", default=None, type=str, help="Path to a object used as the reference for RMSE computation")
parser.add_argument("--distr_path",default=None, type=str, help="If not None saves the rmse distribution as csv file in this location.")
parser.add_argument("--result_dir", default=None, type=str, help="Directory where output results will be stored.")
parser.add_argument("--PCs", default=None, type=str, help="Principal Components to use in the analysis (comma-separated).")
parser.add_argument("--result_name", default=None, type=str, help="Name of the graph.")

def main(args: argparse.Namespace):
    if(args.data_location[-4:]==".csv"):
        rmse_dist = pd.read_csv(args.data_location)
    else:
        paths = helper.extract_pickle_paths(args.data_location)
        if(args.ref_obj_path in paths):
            paths.remove(args.ref_obj_path)
        PCs = list(map(int,args.PCs.split(",")))
        rmse_dist = helper.calculate_rmse_distr_for_sample(paths,PCs,args.ref_obj_path,"Sample_1")
        if(args.distr_path != None):
            rmse_dist.to_csv(args.distr_path)
    p = plotting.ggplot_heatmap_param(rmse_dist)
    p.save(filename=f"{args.result_dir}/{args.result_name}", width=8, height=8, dpi=300)
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)