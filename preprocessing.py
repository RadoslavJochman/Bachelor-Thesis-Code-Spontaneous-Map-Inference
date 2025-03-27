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
import elephant.spike_train_synchrony as sts


def LFP(path):
    """
    Load raw recording from ns6/nix file and extract LFP signal.
    :param path:
    :return:
    """
    if "ns6" in path or "ns5" in path:
        bl = BlackrockIO(path).read_block()
    elif "nix" in path:
        bl = NixIO(path, "ro").read_block()
    seg = bl.segments[0]

    ansignals = []
    for i in range(seg.analogsignals[0].shape[1]):
        anasig = seg.analogsignals[0][:, i]
        anasig = anasig.reshape(anasig.shape[0])
        # 2. Filter the signal between 1Hz and 150Hz
        anasig = butter(anasig, lowpass_frequency=150.0 * pq.Hz)

        # 3. Downsample signal from 30kHz to 500Hz resolution (factor 60)
        anasig = anasig.downsample(30, ftype='fir')

        # 4. Bandstop filter the 50, 100 and 150 Hz frequencies
        # Compensates for artifacts from the European electric grid
        for fq in [50, 100, 150]:
            anasig = butter(anasig, highpass_frequency=(fq + 2) * pq.Hz, lowpass_frequency=(fq - 2) * pq.Hz)
        # seg.analogsignals[0][:, i] = anasig
        ansignals.append(anasig)
    ansignals = np.array(ansignals)
    ansignals = ansignals.reshape((ansignals.shape[0], ansignals.shape[1])).T
    seg = Segment()
    seg.analogsignals.append(AnalogSignal(np.array(ansignals).T, units=pq.mV, sampling_rate=1000 * pq.Hz))
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
    iterator = seg.analogsignals if tag == 'human' else seg.analogsignals[0]
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