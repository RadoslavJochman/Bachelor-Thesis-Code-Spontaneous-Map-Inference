"""
Helper
This script contains functions for
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

import neo
import numpy as np
import os

import yaml
import quantities as pq
import preprocessing
from array_analysis import ArrayAnalysis, METHODS
import pandas as pd
from blackrock_utilities.brpylib             import NevFile, NsxFile, brpylib_ver
from neo import  AnalogSignal
from quantities import  Hz

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

def load_human_segments(data_location: str, layout_path: str, **kwargs):
    """
    Loads and processes neural recording segments from multiple file types (NIX, NS6, NS5, NS2).

    Parameters:
        data_location (str):
            Path to either a directory of data files or a text file listing data paths.
        layout_path (str):
            Path to a CSV file containing a layout with a "chn" column indicating valid channels.
        **kwargs:
            threshold_factor (float):
                Threshold factor for event detection. Required by `preprocessing.nLFP`.
            filter (dict):
                Dictionary specifying filters or filtering parameters used in `preprocessing.nLFP`.
            subtract_mean (bool):
                If True, subtracts the mean (over time) from each channel in each segment.
            z_score (bool):
                If True, applies z-score normalization to segments before mean subtraction.

    Returns:
        list of neo.Segment:
            A list of Neo Segment objects, each containing loaded and processed analog signals.
            All segments share a consistent format for further analysis.
     """

    #Generate paths from directory or text file
    paths = extract_paths(data_location)
    # Read layout, get number of channels
    layout = pd.read_csv(layout_path)
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

    if 'z_score' in kwargs.keys() and kwargs['z_score']:
        segments = preprocessing.zscore_segments(segments)

    # subtract signal mean of from all channels
    if 'subtract_mean' in kwargs.keys() and kwargs['subtract_mean']:
        segments = segments_subtract_mean(segments)
        segments = [preprocessing.nLFP(segment, kwargs['threshold_factor'], kwargs['filter'],tag="human").segments[0]
                    for segment in segments]
    else:
        segments = [preprocessing.nLFP(segment, kwargs['threshold_factor'], kwargs['filter'],tag="human").segments[0]
                    for segment in segments]

    return segments


def get_coords_from_electrode_human(electrode_number):

    electrodes_channels_coords_map = pd.read_pickle('/home/rado/School/bachelor/prosthesis/metadata/electrode_mapping.pickle')
    x = electrodes_channels_coords_map[electrodes_channels_coords_map['New Electrodes']==electrode_number].x.values[0]
    y = electrodes_channels_coords_map[electrodes_channels_coords_map['New Electrodes']==electrode_number].y.values[0]
    return x,y

def zscore_segments(segments):
    from scipy.stats import zscore
    for segment in segments:
        ansigs = segment.analogsignals[0]
        segment.analogsignals[0] = neo.AnalogSignal(zscore(ansigs.magnitude, axis=0), units=ansigs.units,
                                                    t_stop=ansigs.t_stop, sampling_rate=ansigs.sampling_rate)
    return segments

def extract_paths(data_location: str):
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
            paths = [p.strip() for p in paths]
    else:
        paths = os.listdir(data_location)
        paths = [f'{data_location}/{name}' for name in paths]
    return paths