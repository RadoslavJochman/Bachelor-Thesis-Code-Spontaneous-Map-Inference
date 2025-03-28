"""
Author: Radoslav Jochman
"""

import os
import argparse
from array_analysis import load_object
import plotting
from helper import extract_paths

parser = argparse.ArgumentParser(description="Generate spontaneous map graphs using array objects and a reference object with specified principal components.")
parser.add_argument("--array_obj_dir", default=None, type=str, help="Directory path containing the array object(s) to be processed.")
parser.add_argument("--ref_path", default=None, type=str, help="File path to the reference array object used for alignment.")
parser.add_argument("--PCs", default=None, type=str, help="Comma-separated principal components for analysis (e.g., '3,4').")
parser.add_argument("--result_dir", default=None, type=str, help="Directory where the generated graph will be saved.")

def main(args):
    objects_paths = extract_paths(args.array_obj_dir)
    PC1, PC2 = map(int, args.PCs.split(","))
    ref_obj = load_object(args.ref_path)
    ref_obj.compute_new_PC(PC1, PC2)
    for path in objects_paths:
        file_name = os.path.splitext(os.path.basename(path))[0]
        arr_obj = load_object(path)
        arr_obj.compute_new_PC(PC1, PC2)
        p = plotting.ggplot_spontaneous_map(arr_obj, ref_obj)
        result_path = f"{args.result_dir}/spont_{file_name}"
        p.save(filename=result_path, width=8, height=8, dpi=300)

if __name__ == '__main__':
    main_args = parser.parse_args()
    main(main_args)