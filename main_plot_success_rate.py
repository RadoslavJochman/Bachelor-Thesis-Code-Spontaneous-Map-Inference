import plotting
import helper
import argparse
import pickle
import pandas as pd
from array_analysis import load_object
import numpy as np
from helper import distance_of_patterns_in_map, distance_in_space, get_TH
parser = argparse.ArgumentParser()
parser.add_argument("--analysis_obj_path", default=None, type=str, help="Path to the analysis object file")
parser.add_argument("--PCs", default=None, type=str, help="Comma-separated principal components for analysis (e.g., '3,4').")
parser.add_argument("--res_dir", default=None, type=str, help="Directory path where the resulting plot will be saved.")
parser.add_argument("--res_name", default=None, type=str, help="Filename for the resulting plot.")





def main(args):
    PC1, PC2 = map(int, args.PCs.split(","))
    ptp = 'metadata/discrimination_task_patient4.pkl'
    with open(ptp, "rb") as resfile:
        perc_data = pickle.load(resfile)
    arr_obj = load_object(args.analysis_obj_path)
    arr_obj.compute_new_PC(PC1, PC2)
    success_measure = 'correct'
    suc_rates = []
    sdists = []
    mdists = []
    patterns_list = []
    for i in range(len(perc_data)):
        try:
            patterns = [perc_data[i]['electrodes_1'], perc_data[i]['electrodes_2']]
            correct = perc_data[i][success_measure]
            nr_answers = perc_data[i]['nr_answers']
            #delays = perc_data[i]['delays']
        except Exception as e:
            print(e)
            print('Problem, skipping pattern.')
            continue

        mdist = np.array(distance_of_patterns_in_map(patterns[0], patterns[1], arr_obj))
        sdist = np.array(distance_in_space(patterns[0], patterns[1], arr_obj))
        patterns_list.append(patterns)

        if len(sdist) == 0:
            print('No dists, probably grey region.')
            continue

        mdists.append(mdist.std() / mdist.mean())
        sdists.append(sdist.std() / sdist.mean())
        suc_rates.append(np.sum(correct) / nr_answers)
    df = pd.DataFrame({
        "functional_distance": mdists,
        "spatial_distance": sdists,
        "success_rate": suc_rates
    })
    p = plotting.ggplot_success_rate(df)
    result_path = f"{args.res_dir}/{args.res_name}"
    p.save(filename=result_path, width=8, height=8, dpi=300)


if __name__ == '__main__':
    main_args = parser.parse_args()
    main(main_args)