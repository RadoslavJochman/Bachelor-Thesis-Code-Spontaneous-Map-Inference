import argparse
import plotting
import pandas as pd
from plotnine import theme, element_rect
from analysis import compute_spontaneous_map
from array_analysis import ArrayAnalysis, METHODS
from helper import load_human_segments, split_segment
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--analysis_params_path", default=None, type=str, help="Path to the YAML file that defines the parameters for the analysis process.")
parser.add_argument("--params_path", default=None, type=str, help="Path to the YAML file containing configuration parameters for data loading and preprocessing.")
parser.add_argument("--data_location", default=None, type=str, help="Path to a directory with data or a text file listing paths to individual data files.")
parser.add_argument("--result_dir", default=None, type=str, help="Directory where output results will be stored.")
parser.add_argument("--result_name", default=None, type=str, help="Name of the graph.")
parser.add_argument("--PCs", default=None, type=str, help="Principal Components to use in the analysis (comma-separated).")
parser.add_argument("--segment_size", default=None, type=int, help="Duration in seconds for each data segment used during the analysis.")

def main(args: argparse.Namespace):
    with open(args.analysis_params_path) as f:
        analysis_params = yaml.safe_load(f)

    with open(args.params_path) as f:
        params = yaml.safe_load(f)
    layout = pd.read_csv(params["layout_path"])
    segments = load_human_segments(args.data_location,params)
    arr_objs = []
    for seg in segments:
        arr_objs.extend([ArrayAnalysis("4",METHODS.nLFP,[split_seg],layout,analysis_params) for split_seg in split_segment(seg,args.segment_size)])

    PC1, PC2 = map(int,args.PCs.split(","))
    for obj in arr_objs:
        compute_spontaneous_map(obj)
        obj.compute_new_PC(PC1, PC2)
    ref_obj = arr_objs[0]
    rmse = {}
    for i,obj in enumerate(arr_objs):
        rot, _ = plotting.find_ideal_rotation(ref_obj.spontaneous_map,obj.spontaneous_map)
        rmse[f"{i}"] = plotting.rmse_angles(rot,ref_obj.spontaneous_map)
    p = plotting.ggplot_rmse(rmse,ref_obj)
    p = p + theme(
        plot_background=element_rect(fill="white", color="white"),
        panel_background=element_rect(fill="white", color="white")
    )
    p.save(filename=f"{args.result_dir}/{args.result_name}", width=10, height=6, dpi=300)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)