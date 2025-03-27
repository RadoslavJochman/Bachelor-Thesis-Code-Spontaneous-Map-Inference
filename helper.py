"""
Helper
This script contains functions for
subtracting mean across all channels from segments
    -segments_subtract_mean()
loading human data from nnx file, and preprocess it
    -load_human_data()
getting coords from electrode index
    -get_coords_from_electrode_human()
Calculate standard score for segments
    -zscore_segments()
Authors: Karolína Korvasová, Matěj Voldřich
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
    Subtract mean from channel analog signals for each segment.
    """
    for segment in segments:
        analog_signal = segment.analogsignals[0]
        segment.analogsignals[0] = (analog_signal.T - np.mean(analog_signal, axis=1)).T
    return segments


def load_human_data(patient_id: str, data_location: str, layout_path: str, **kwargs):
    # We accommodate the option to just load data from a single folder, and also the
    # option to load data from folder paths stored in a text file
    if os.path.isfile(data_location):
        with open(data_location) as f:
            paths = f.readlines()
            paths = [p.strip() for p in paths]
    else:
        paths = os.listdir(data_location)
        paths = [f'{data_location}/{name}' for name in paths]

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
            print(path)
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
            s.analogsignals = [AnalogSignal(raw[:n_channels], units="uV", sampling_rate=1000 * Hz).rescale('mV')]
            segments.append(s)

    if 'z_score' in kwargs.keys() and kwargs['z_score']:
        segments = preprocessing.zscore_segments(segments)

    # subtract signal mean of from all channels
    if 'subtract_mean' in kwargs.keys() and kwargs['subtract_mean']:
        segments = segments_subtract_mean(segments)
        segments = [preprocessing.nLFP(segment, kwargs['threshold_factor'], kwargs['filter']).segments[0]
                    for segment in segments]
    else:
        segments = [preprocessing.nLFP(segment, kwargs['threshold_factor'], kwargs['filter']).segments[0]
                    for segment in segments]

    # load electrode layout (and params)
    params = None
    if 'params_path' in kwargs.keys():
        with open(kwargs['params_path']) as f:
            params = yaml.safe_load(f)

    arr_obj = ArrayAnalysis(f"{patient_id}_nLFP", input_type=METHODS.nLFP,
                            segments=segments, layout=layout, params=params)
    return arr_obj

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