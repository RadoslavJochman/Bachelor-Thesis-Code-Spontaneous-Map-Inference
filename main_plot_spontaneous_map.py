"""
Script: main_plot_spontaneous_map.py

Author: Radoslav Jochman

Description:
    This script generates spontaneous map visualizations by comparing array objects
    to a reference array object using specified principal components (PCs). It loads
    array objects from a directory, aligns them with a reference object's spontaneous
    map, and plot spontaneous map. The plots are saved to the designated output directory.

Usage:
    python main_plot_spontaneous_map.py
        --array_obj_dir path/to/array_objects_directory
        --ref_path path/to/reference_object.pkl
        --PCs 3,4
        --result_dir path/to/output_directory

Arguments:
    --array_obj_dir : str
        Directory containing pickled array objects to process.

    --ref_path : str
        Path to the pickled reference array object for alignment.

    --PCs : str
        Comma-separated principal components to use for the analysis (e.g., "3,4").

    --result_dir : str
        Directory where the resulting spontaneous map plots will be saved.

Output:
    - PNG images showing spontaneous map comparisons between each array object
      and the reference object, saved under the specified result directory.

Requirements:
    - The `array_analysis` module must provide:
        - `load_object(path) -> array_object`
        - `array_object.compute_new_PC(PC1, PC2)`
    - Input files must be pickled array objects with spontaneous maps computed.
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
    ref_sample = args.ref_path.split("/")[-2]
    ref_obj = load_object(args.ref_path)
    ref_obj.compute_new_PC(PC1, PC2)
    for path in objects_paths:
        file_name = os.path.splitext(os.path.basename(path))[0]
        sample = path.split("/")[-2][7:]
        arr_obj = load_object(path)
        arr_obj.compute_new_PC(PC1, PC2)
        p = plotting.ggplot_spontaneous_map_human(arr_obj, ref_obj)
        result_path = f"{args.result_dir}/ref_{ref_sample}_{sample}_spont_{file_name}.png"
        p.save(filename=result_path, width=8, height=8, dpi=300)

if __name__ == '__main__':
    main_args = parser.parse_args()
    main(main_args)