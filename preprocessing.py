"""
Preprocessing
This script contains functions for
Extracting LFP from raw signal
    -LFP()
Detecting nLFP from LFP:
    -nLFP()
Authors: Karolína Korvasová, Matěj Voldřich
"""

import neo
from elephant.signal_processing import butter
import quantities as pq
import elephant
import numpy as np
from neo import NixIO, BlackrockIO, Segment, AnalogSignal, Block


def LFP(path):
    """
    Loads a raw neural recording file (ns5/ns6/NIX) and extracts the LFP (Local Field Potential) signal.

    This function performs several processing steps:
      1. Reads a neural data file using `BlackrockIO` (for ns5/ns6) or `NixIO` (for NIX format).
      2. Retrieves the first segment and extracts the AnalogSignal.
      3. Applies a low-pass filter (150 Hz), then downsamples the signal from 30 kHz to 500 Hz.
      4. Removes line noise at 50 Hz, 100 Hz, and 150 Hz using bandstop filters.
      5. Constructs a `Block` containing a single `Segment` with the processed AnalogSignal.

    Parameters:
        path (str):
            Path to the data file. Supported formats:
              - Blackrock ns5 or ns6
              - NIX (.nix)

    Returns:
        neo.Block:
            A Neo `Block` object containing:
              - A single `Segment` with one downsampled, filtered `AnalogSignal` in mV,
                sampled at 1000 Hz with shape (n_timepoints, n_channels)
    """
    if "ns6" in path or "ns5" in path:
        bl = BlackrockIO(path).read_block()
    elif "nix" in path:
        bl = NixIO(path, "ro").read_block()
    seg = bl.segments[0]

    anasignals = []
    for i in range(seg.analogsignals[0].shape[1]):
        anasig = seg.analogsignals[0][:, i]
        anasig = anasig.reshape(anasig.shape[0])
        # 2. Filter the signal between 1Hz and 150Hz
        anasig = butter(anasig, lowpass_frequency=150.0 * pq.Hz)

        # 3. Downsample signal from 30kHz to 1000Hz resolution (factor 30)
        anasig = anasig.downsample(30, ftype='fir')

        # 4. Bandstop filter the 50, 100 and 150 Hz frequencies
        # Compensates for artifacts from the European electric grid
        for fq in [50, 100, 150]:
            anasig = butter(anasig, highpass_frequency=(fq + 2) * pq.Hz, lowpass_frequency=(fq - 2) * pq.Hz)
        anasignals.append(anasig.rescale('mV'))
    anasignals = np.array(anasignals)
    anasignals = anasignals.reshape((anasignals.shape[0], anasignals.shape[1])).T
    seg = Segment()
    seg.analogsignals.append(AnalogSignal(np.array(anasignals), units=pq.mV, sampling_rate=1000 * pq.Hz))
    bl = Block()
    bl.segments.append(seg)
    return bl

def nLFP(LFP_signal, thr_factor, filter, tag='monkey'):
    """
    Load LFP recording from ns6/nix file and detect nLFP events.
    :param tag:
    :param LFP_signal: path to LFP recording or LFP neo.Segment
    :param thr_factor:
    :param filter: bandpass filter frequencies
    :return:
    """
    if type(LFP_signal) is str:
        bl = neo.NixIO(LFP_signal, "ro").read_block()
        seg = bl.segments[0]
    else:
        seg = LFP_signal

    bl = neo.Block()
    bl.segments.append(seg)

    before_art_removal = []
    trials = [[seg.analogsignals[0].t_start,
               seg.analogsignals[0].t_stop]]
    iterator = seg.analogsignals if tag == 'human' else seg.analogsignals[0].T
    for anasig in iterator:
        anasig = elephant.signal_processing.butter(anasig,
                                                   lowpass_frequency=filter[1] * pq.Hz,
                                                   highpass_frequency=filter[0] * pq.Hz,
                                                   sampling_frequency=seg.analogsignals[0].sampling_rate)
        for tnr, trial in enumerate(trials):
            if tnr == 0:  # define the threshold based on the spontaneous activity
                thr = np.mean(anasig) + thr_factor * np.std(anasig)
                if thr_factor > 0:
                    sign_ext = 'above'
                else:
                    sign_ext = 'below'

            # extract peaks
            anasig = AnalogSignal(np.array(anasig), units=pq.mV, sampling_rate=1000 * pq.Hz)
            st = elephant.spike_train_generation.peak_detection(
                anasig,
                threshold=np.array(thr) * anasig.units,
                sign=sign_ext)
            bl.segments[tnr].spiketrains.append(st)
            before_art_removal.append(len(st))
    return bl