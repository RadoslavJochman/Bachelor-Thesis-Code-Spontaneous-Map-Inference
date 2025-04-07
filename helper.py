"""
Helper
This script contains functions for
Aligning spontaneous map to a reference map
    -find_ideal_rotation()
Calculating circular difference between two spontaneous maps
    -circ_diff()
Calculating RMSE of two spontaneous maps
    -rmse_angles()
Generating control distribution of RMSE and calculating percentile
    -generate_control_rmse_distr()
subtracting mean across all channels from segments
    -segments_subtract_mean()
loading human data from nnx file, and preprocess it
    -load_human_segments()
getting coords from electrode index
    -get_coords_from_electrode_human()
Calculate standard score for segments
    -zscore_segments()
Authors: Karolína Korvasová, Matěj Voldřich
Modifications by: Radoslav Jochman
"""
import itertools
from argparse import ArgumentError
import neo
import numpy as np
from array_analysis import ArrayAnalysis, load_object
import os
import preprocessing
import pandas as pd
from blackrock_utilities.brpylib             import NevFile, NsxFile, brpylib_ver
from quantities import  Hz

def distance_in_space(pattern1, pattern2, array_obj: ArrayAnalysis):
    '''
    Calculate pairwise spatial distances of electrodes in two electrodes sets (patterns).
    :param pattern1: list of electrode numbers
    :param pattern2: list of electrode numbers
    :return: list of spatial distances
    '''

    resob =  array_obj

    dists = []
    for el1 in pattern1:
        for el2 in pattern2:
            chnix = convert_electrode_to_channel_human(el1) - 1
            cx1, cy1 = resob.get_channel_position(int(chnix))
            col1 = resob.spontaneous_values[int(chnix)]

            chnix = convert_electrode_to_channel_human(el2) - 1
            cx2, cy2 = resob.get_channel_position(int(chnix))
            col2 = resob.spontaneous_values[int(chnix)]

            if (col1 > -0.5) and (col2 > -0.5):
                dists.append(np.linalg.norm(np.array([cx1*0.4, cy1*0.4])-np.array([cx2*0.4, cy2*0.4]), ord=2))

    return dists

def distance_of_patterns_in_map(pattern1, pattern2, array_obj: ArrayAnalysis):
    '''
    Calculate pairwise functional distances of electrodes in two electrodes sets (patterns).
    :param pattern1: list of electrode numbers
    :param pattern2: list of electrode numbers
    :return: list of functional distances
    '''


    map_vals = array_obj.spontaneous_values

    colors1 = []
    for el in pattern1:

        chnix = convert_electrode_to_channel_human(el)-1
        colors1.append(map_vals[int(chnix)])

    colors2 = []
    for el in pattern2:

        chnix = convert_electrode_to_channel_human(el)-1
        colors2.append(map_vals[int(chnix)])

    #check that colors are in radians
    assert max(colors1+colors2)<4

    dists = []
    for i, col1 in enumerate(colors1):
        for col2 in colors2:
            if (col1>-0.5) and (col2>-0.5):
                dists.append(np.min((np.abs(col1 - col2), 3.14 - np.abs(col1 - col2))))

    return dists

def join_dataframe(*args):
    result = args[0]
    for dataframe in args[1:]:
        result = pd.concat([result, dataframe], ignore_index=True)
    return result

def find_ideal_rotation(ref: np.ndarray, rot: np.ndarray, n_steps: int=5000):
    """
    Finds the rotation offset (from 0 to π) that best aligns a given spontaneous map to a reference map.

    This function evaluates equally spaced rotation offsets (steps) within [0, π] and applies each offset
    to the map `rot`. It then computes the circular difference between the offset map and the reference map
    `ref` (using mean squared error as a criterion), selecting the offset that yields the smallest MSE.

    Parameters:
        ref (numpy.ndarray):
            The reference spontaneous map (2D array).
        rot (numpy.ndarray):
            The map to be aligned (2D array with same shape as 'ref').
        n_steps (int, optional):
            Number of rotation increments in the search space, evenly distributed from 0 to π.
            A larger value produces finer precision but increases computational cost.

    Returns:
        numpy.ndarray:
            The best-aligned map.
    """
    deleted_mask = rot==-2
    steps = np.linspace(0, np.pi, n_steps)
    best_err = np.inf
    best_rotated = None

    for step in steps:
        candidate = np.mod(rot + step, np.pi)
        diff = circ_diff(ref, candidate)
        err = np.mean(diff ** 2)
        if err < best_err:
            best_err = err
            best_rotated = candidate
    #mark deleted channels as not valid

    best_rotated[deleted_mask] = -2
    return best_rotated

def circ_diff(a, b):
    """
    Computes the circular difference (with a period of π) between two spontaneous maps.

    This function calculates the absolute difference between two angles (or angle arrays),
    then adjusts it to ensure the difference is within the range [0, π/2], effectively
    treating π as a full period.

    Parameters:
        a: float or array-like
            The first angle or collection of angles in radians. Typically in [0, π).
        b: float or array-like
            The second angle or collection of angles in radians. Typically in [0, π).

    Returns:
        float or numpy.ndarray:
            The circular difference between the inputs, within the range [0, π/2].
            If inputs are arrays, the output is an array of the same shape.
    """
    diff = np.abs(a - b)
    return np.minimum(diff, np.pi - diff)

def rmse_angles(a, b):
    """
    Calculates the root mean square error (RMSE) for two angle arrays, accounting for circular difference.

    This function uses `circ_diff` to compute the circular difference (period of π) between corresponding
    angles in `a` and `b`, then calculates the square root of the mean of the squared differences.

    Parameters:
        a (array-like):
            The first set of angles in radians.
        b (array-like):
            The second set of angles in radians, typically matching `a`.

    Returns:
        float:
            The RMSE value representing the circular difference between `a` and `b`.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    deleted_mask = (a < 0) | (b < 0)
    a[deleted_mask] = np.nan
    b[deleted_mask] = np.nan
    differences = circ_diff(a, b)
    mse = np.nanmean(differences**2, where=differences!=np.nan)
    return np.sqrt(mse)

def generate_control_rmse_distr(ref_map: np.ndarray, n_iter: int=2000, percentile: int=1):
    """
    Creates a distribution of RMSE values by randomly permuting a reference map and aligning each permutation
    back to the original.

    The function performs the following steps for each iteration:
      1. Flattens `ref_map` and randomly permutes its elements, then reshapes it to the original shape.
      2. Aligns the permuted map to the reference map using `find_ideal_rotation`.
      3. Computes the RMSE between the reference map and the aligned permutation using `rmse_angles`.
      4. Collects these RMSE values into a list.

    After all iterations, the function returns the specified percentile value of the resulting RMSE distribution.
    This process can be used to estimate a baseline or threshold for statistical analysis when comparing map
    alignments.

    Parameters:
        ref_map (numpy.ndarray):
            The reference spontaneous map, typically a 2D array.
        n_iter (int, optional):
            Number of random permutations to generate. Default is 2000.
        percentile (float, optional):
            Which percentile of the RMSE distribution to return. Default is 1 (1st percentile).

    Returns:
        float:
            The selected percentile from the distribution of RMSE values obtained from random permutations.
    """
    rmse_dist = []
    for i in range(n_iter):
        permut_map = np.random.permutation(ref_map.flatten()).reshape(ref_map.shape)
        permut_map = find_ideal_rotation(ref_map,permut_map)
        rmse = rmse_angles(ref_map,permut_map)
        rmse_dist.append(rmse)
    return np.percentile(np.array(rmse_dist), percentile)

def segments_subtract_mean(segments):
    """
    Applies Common Average Referencing (CAR) across channels for each segment at each time point.

    Parameters:
        segments (list of neo.Segment):
            A list of Neo Segment objects. This data should be in the shape
            (n_channels, n_timepoints)

    Returns:
        list of neo.Segment:
            The same list of segments, but with each segment's analog signals updated to have
            zero mean across channels at each time point.
    """
    for segment in segments:
        analog_signal = np.array(segment.analogsignals).squeeze()
        segment.analogsignals = analog_signal - np.mean(analog_signal,axis=0)
    return segments

def load_human_segments(data_location: str, params: dict):
    """
    Loads and processes neural recording segments from multiple file types (NIX, NS6, NS5, NS2).

    Parameters:
        data_location (str):
            Path to a directory containing data files or a text file listing the paths to data files.
        params (dict):
            Dictionary of additional processing parameters. Expected keys include:
              - threshold_factor (float): Factor for event detection used by preprocessing.nLFP.
              - filter (dict): Dictionary specifying filters or filtering parameters for preprocessing.nLFP.
              - subtract_mean (bool): If True, subtracts the mean (over time) from each channel in the segments.
              - z_score (bool): If True, applies z-score normalization to the segments prior to mean subtraction.
              - layout_path (str): Path to a CSV file with layout information, including a "chn" column indicating valid channels.

    Returns:
        list of neo.Segment:
            A list of neo.Segment objects, each containing loaded and processed analog signals formatted
            consistently for further analysis.
    """

    #Generate paths from directory or text file
    paths = extract_paths(data_location)
    # Read layout, get number of channels
    layout = pd.read_csv(params["layout_path"])
    n_channels = np.sum(~np.isnan(layout["chn"]))

    # The nix file that the Prague lab makes is configured to have in each segment  n_channels
    # separate analogsignals of length recording_time

    # The ns2 file made by the SW in ELche lab has segments with one or two analogsignals. One
    # of these is of the shape n_channels x recording_time, the other one if it is there is of
    # the shape 2 x recording_time and it is the audio recording from the experiment

    # To not have to change the further computations, we are restructuring the ns2 semgents
    # such that they have the same format as the .nix segments
    segments = []
    segments.extend([neo.NixIO(path, "ro").read_block().segments[0] for path in paths if '.nix' in path])

    for path in paths:
        if ".ns6" in path or ".ns5" in path:
            s = preprocessing.LFP(path).segments[0]
            segments.append(s)
        elif '.ns2' in path:
            s = neo.Segment()
            # load RAW DATA
            nsx_file = NsxFile(path)
            raw = nsx_file.getdata('all')
            nsx_file.close()
            del nsx_file
            raw = raw['data']
            s.analogsignals = [neo.AnalogSignal(raw[i, :], units="uV", sampling_rate=1000 * Hz).rescale('mV') for i in
                               range(n_channels)]
            segments.append(s)

    return segments

def process_LFP_to_nLFP(segments: list, params: dict):
    if 'z_score' in params.keys() and params['z_score']:
        segments = zscore_segments(segments)

    # subtract signal mean of from all channels
    if 'subtract_mean' in params.keys() and params['subtract_mean']:
        segments = segments_subtract_mean(segments)
        segments = [preprocessing.nLFP(segment, params['threshold_factor'], params['filter'],tag="human").segments[0]
                    for segment in segments]
    else:
        segments = [preprocessing.nLFP(segment, params['threshold_factor'], params['filter'],tag="human").segments[0]
                    for segment in segments]
    return segments
def get_coords_from_electrode_human(electrode_number):

    electrodes_channels_coords_map = pd.read_pickle('metadata/electrode_mapping.pickle')
    x = electrodes_channels_coords_map[electrodes_channels_coords_map['New Electrodes']==electrode_number].x.values[0]
    y = electrodes_channels_coords_map[electrodes_channels_coords_map['New Electrodes']==electrode_number].y.values[0]
    return x,y

def convert_electrode_to_channel_human(electrode_number):

    electrodes_channels_coords_map = pd.read_pickle('metadata/electrode_mapping.pickle')
    return electrodes_channels_coords_map[electrodes_channels_coords_map['New Electrodes']==electrode_number].Channels.values[0]

def zscore_segments(segments):
    from scipy.stats import zscore
    for segment in segments:
        ansigs = segment.analogsignals[0]
        segment.analogsignals[0] = neo.AnalogSignal(zscore(ansigs.magnitude, axis=0), units=ansigs.units,
                                                    t_stop=ansigs.t_stop, sampling_rate=ansigs.sampling_rate)
    return segments

def extract_paths(data_location: str, extension: str=None):
    """
    Extracts file paths based on the given data location.

    This function can handle two scenarios:
      1. If `data_location` is a text file, it reads each line from the file (assuming each contains one path).
      2. Otherwise, if `data_location` is a directory, it lists all files in that directory and builds
         full paths for each.

    Parameters:
        data_location (str):
            Either:
              - A path to a text file containing one data file path per line,
              - Or a directory containing data files.

    Returns:
        list of str:
            A list of file paths. If `data_location` was a file, each line is treated as a path.
            If `data_location` was a directory, the function returns a list of paths in that directory.
    """
    if os.path.isfile(data_location):
        with open(data_location) as f:
            paths = f.readlines()
            paths = [p.strip() for p in paths if extension is None or extension in p]
    else:
        paths = os.listdir(data_location)
        paths = [f'{data_location}/{name}' for name in paths if ".pkl" in name]
    return paths

def split_segment(segment: neo.Segment, duration_s: int) -> list[neo.Segment]:
    """
    Splits a neo.Segment's analog signals into smaller segments of equal duration.

    This function takes an input neo.Segment and divides its contained analog signals
    into multiple new segments, each spanning 'duration_s' seconds. It operates under the
    assumption that all analog signals within the segment share the same sampling rate,
    units, and total duration.

    Parameters:
        segment (neo.Segment): The input segment containing analog signals to be split.
        duration_s (int): The desired duration (in seconds) for each output segment.

    Returns:
        list[neo.Segment]: A list of new neo.Segment objects, each holding a subset of the
        original analog signals corresponding to the specified segment duration.
    """

    segments = []
    ansigs = segment.analogsignals
    fq = segment.analogsignals[0].sampling_rate
    units = segment.analogsignals[0].units
    duration_time_s = segment.analogsignals[0].t_stop
    duration_time_s.units = "s"
    num_samples = int(duration_time_s.item() // duration_s)
    duration_num_values = segment.analogsignals[0].shape[0]
    duration_per_sample = int(duration_num_values//num_samples)
    for i in range(num_samples):
        s = neo.Segment()
        s.analogsignals = [neo.AnalogSignal(ansigs[j][i*duration_per_sample:(i+1)*duration_per_sample,:], units=units, sampling_rate=fq) for j in range(len(segment.analogsignals))]
        segments.append(s)
    return segments

def calculate_rmse_dist_for_pcs(arr_obj:ArrayAnalysis, ref_obj:ArrayAnalysis, PCs: list ):
    """
    Computes the RMSE between spontaneous maps for each unique pair of principal components (PCs) from two ArrayAnalysis objects.

    Parameters:
          arr_obj (ArrayAnalysis): The analysis object for which the spontaneous map is computed using each PC pair.
          ref_obj (ArrayAnalysis): The reference analysis object for which the spontaneous map is computed using each PC pair.
          PCs (list): A list of principal component; at least two must be provided.

    Returns:
      pd.DataFrame: A DataFrame with columns "PCs" (PC pair as a string, e.g., "3,4") and "rmse" (the computed RMSE value for that pair).

    Raises:
      ArgumentError: If fewer than 2 principal components are provided in the PCs list.
    """
    if len(PCs)<2:
        raise ArgumentError(message="Need at least 2 PCs.")
    result_PCs = []
    result_rmse = []
    result = pd.DataFrame()
    for PC1 in PCs:
        for PC2 in PCs:
            if PC1<PC2:
                arr_obj.compute_new_PC(PC1, PC2)
                ref_obj.compute_new_PC(PC1, PC2)
                ref_map = ref_obj.spontaneous_map
                arr_map = arr_obj.spontaneous_map
                arr_map = find_ideal_rotation(ref_map,arr_map)
                result_PCs.append(f"{PC1},{PC2}")
                result_rmse.append(rmse_angles(ref_map,arr_map))
    result["PCs"] = result_PCs
    result["rmse"] = result_rmse
    return result

def filter_paths_by_TH(paths: list[str], TH: int):
    """Return file paths whose filename contains the threshold value (as string) immediately following 'fac'."""
    filtered_paths = []
    for path in paths:
        if(get_TH(path)!=None and get_TH(path)==str(TH)):
            filtered_paths.append(path)
    return filtered_paths

def filter_paths_by_bin_size(paths: list[str], bin_size: int):
    """Return file paths whose filename contains the bin size (as string) immediately following 'bin_size'."""
    filtered_paths = []
    for path in paths:
        if(get_bin_size(path)!=None and get_bin_size(path)==str(bin_size)):
            filtered_paths.append(path)
    return filtered_paths

def get_TH(path):
    """Extract and return the threshold value from the filename after 'fac', or None if not found."""
    file_name = os.path.basename(path)
    split_name = file_name.split("_")
    if ("fac" in split_name):
        th_index = split_name.index("fac") + 1
        return split_name[th_index]
    else: return None

def get_bin_size(path):
    """Extract and return the bin size value from the filename after 'size', or None if not found."""
    file_name = os.path.basename(path)
    split_name = file_name.split("_")
    if ("size" in split_name):
        bin_index = split_name.index("size") + 1
        bin_size = ".".join(split_name[bin_index].split(".")[0:2])
        return bin_size
    else: return None

def calculate_rmse_distr_for_sample(paths: list[str],PCs: list[int], ref_obj_path, sample: str):
    """
    Computes the RMSE between the spontaneous maps of sample objects and a fixed reference object
    across multiple principal component (PC) pairs.

    This function processes a collection of sample objects (provided as file paths) by comparing their
    spontaneous maps with that of a reference object loaded from `ref_obj_path`. For every unique PC pair
    (using combinations from the sorted list of PCs), the function:
      - Updates the reference object's spontaneous map using the current PC pair by calling compute_new_PC.
      - Iterates over each sample file in `paths`:
          * Extracts the threshold (TH) and bin size information from the filename using helper functions
            get_TH and get_bin_size.
          * Loads the sample object and recalculates its spontaneous map with the same PC pair.
          * Aligns the sample object's spontaneous map to the reference object's map using find_ideal_rotation.
          * Computes the RMSE between the aligned sample map and the reference map via rmse_angles.
          * Records the sample identifier, extracted bin size, formatted PC pair (e.g., "3,4"), threshold,
            and the computed RMSE in a results list.

    Finally, the function aggregates the results into a pandas DataFrame with the following columns:
      - "sample": Identifier for the sample.
      - "bin_size": The bin size extracted from the sample file name.
      - "PC_pair": A string representation of the PC pair (formatted as "PC1,PC2").
      - "TH": The threshold value extracted from the sample file name.
      - "RMSE": The computed root mean square error between the reference and sample spontaneous maps.

    Parameters:
      paths (list[str]): List of file paths for the sample objects.
      PCs (list[int]): List of principal component indices to evaluate; must contain at least two values.
      ref_obj_path: File path to the reference object used to generate the baseline spontaneous map.
      sample (str): Identifier for the sample, to be included in the result DataFrame.

    Returns:
      pd.DataFrame: A DataFrame summarizing the RMSE values for each PC pair and sample object, with columns:
                    ["sample", "bin_size", "PC_pair", "TH", "RMSE"].
    """
    ref_obj = load_object(ref_obj_path)
    PCs.sort()
    results_list = []
    for PC1,PC2 in itertools.combinations(PCs,2):
        ref_obj.compute_new_PC(PC1, PC2)
        for path in paths:
            TH = get_TH(path)
            bin_size = get_bin_size(path)
            arr_obj = load_object(path)
            arr_obj.compute_new_PC(PC1, PC2)
            arr_map = find_ideal_rotation(ref_obj.spontaneous_map,arr_obj.spontaneous_map)
            rmse = rmse_angles(arr_map,ref_obj.spontaneous_map)
            results_list.append([sample,bin_size,f"{PC1},{PC2}",TH,rmse])
    result = pd.DataFrame(results_list,columns=["sample","bin_size","PC_pair", "TH", "RMSE"])
    return result

def calculate_rmse_distr_sample_and_ref_sample(paths: list[str],PCs: list[int], ref_sample_paths: list[str], sample: str):
    """
    Computes the RMSE between spontaneous maps for paired sample and reference objects across multiple principal component (PC) combinations.

    The function processes corresponding file pairs from `paths` (sample objects) and `ref_sample_paths` (reference objects) after sorting them.
    For each pair, it:
      - Extracts the threshold (TH) and bin size from the filename using helper functions `get_TH` and `get_bin_size`.
      - Verifies that the TH and bin size values in the sample and reference filenames match; if not, it raises an ArgumentError.
      - Loads both the sample and reference objects using `load_object`.
      - Iterates over every unique combination of two PCs (using itertools.combinations on the sorted `PCs` list), where for each PC pair (PC1, PC2):
          * Recalculates the spontaneous maps for both objects by invoking `compute_new_PC(PC1, PC2)`.
          * Aligns the sample object's spontaneous map to the reference object's spontaneous map with `find_ideal_rotation`.
          * Computes the RMSE between the aligned sample map and the reference map via `rmse_angles`.
          * Records a result row containing the sample identifier, bin size, formatted PC pair (as "PC1,PC2"), TH, and the RMSE value.

    Parameters:
      paths (list[str]): List of file paths for the sample objects.
      PCs (list[int]): List of principal component indices to evaluate; at least two are required.
      ref_sample_paths (list[str]): List of file paths for the corresponding reference objects.
      sample (str): Identifier for the sample, which will be recorded in the results.

    Returns:
      pd.DataFrame: A DataFrame with columns ["sample", "bin_size", "PC_pair", "TH", "RMSE"], where each row corresponds to
      the evaluation of a unique PC pair for a sample-reference object pair.

    Raises:
      ArgumentError: If the number of sample objects does not match the number of reference objects, or if the extracted TH or
      bin size from a sample file does not match that of its corresponding reference file.
    """
    results_list =[]
    paths.sort()
    ref_sample_paths.sort()
    PCs.sort()
    if(len(ref_sample_paths)!=len(paths)):
        raise ArgumentError(message="The number of reference objects has to be the same as number of processed objects.")
    for ref_path, path in zip(ref_sample_paths, paths):
        TH = get_TH(path)
        bin_size = get_bin_size(path)
        TH_ref = get_TH(ref_path)
        bin_size_ref = get_bin_size(ref_path)
        if(TH!=TH_ref or bin_size!=bin_size_ref):
            raise ArgumentError(message="\"paths\" and \"ref_sample_paths\" has to contain pairs of objects named the as TH_fac_*_bin_size_*.pkl")
        ref_obj = load_object(ref_path)
        arr_obj = load_object(path)
        for PC1, PC2 in itertools.combinations(PCs,2):
            arr_obj.compute_new_PC(PC1, PC2)
            ref_obj.compute_new_PC(PC1, PC2)
            arr_map = find_ideal_rotation(ref_obj.spontaneous_map, arr_obj.spontaneous_map)
            rmse = rmse_angles(arr_map, ref_obj.spontaneous_map)
            results_list.append([sample, bin_size, f"{PC1+1},{PC2+1}", TH, rmse])
    result = pd.DataFrame(results_list, columns=["sample", "bin_size", "PC_pair", "TH", "RMSE"])

    return result