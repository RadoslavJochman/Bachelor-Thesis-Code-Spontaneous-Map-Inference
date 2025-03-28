"""
Analysis
This script contains functions for
1) generating frames
    - compute_frames()
2) data interpolation
    - interpolate_data()
3) generating correlation maps
    - compute_correlation_maps()
4) generating spontaneous maps
    - def compute_spontaneous_map()

Authors: Karolína Korvasová, Matěj Voldřich
"""

import scipy.stats
from array_analysis import ArrayAnalysis, METHODS
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr


def compute_frames(analysis_obj: ArrayAnalysis):
    """
    Compute frames for given array analysis object based on the method.

    :param analysis_obj:
    :return:
    """
    frames_all = []
    for segment in analysis_obj.segments:
        if analysis_obj.method == METHODS.tMUA or analysis_obj.method == METHODS.nLFP:
            # calculate frames as binned activity (number of spikes)
            events = segment.spiketrains
            step = analysis_obj.params['event_binsize']
            if analysis_obj.params['overlap_frames']:
                step /= 2
            duration = segment.t_stop.magnitude
            frames = np.zeros((int(duration / step), analysis_obj.n_channels))
            for i in range(frames.shape[0]):
                t1 = i * step
                t2 = (i + 1) * step
                for j in range(frames.shape[1]):
                    frames[i, j] = np.sum(np.logical_and(events[j] >= t1, events[j] < t2))
            if analysis_obj.params['overlap_frames']:
                # sum frame halfs
                for i in range(frames.shape[0]):
                    for j in range(frames.shape[1] - 1):
                        frames[i, j] = np.sum(frames[i, j:j+1])
        elif analysis_obj.method == METHODS.MUAe:
            frames = segment.analogsignals[0]
        elif analysis_obj.method == METHODS.LFP:
            from scipy.signal import welch
            # compute frames as binned total power
            step = analysis_obj.sampling_rate * analysis_obj.params['event_binsize']
            signal = segment.analogsignals[0]
            frames = np.zeros((int(signal.shape[0] / step), analysis_obj.n_channels))
            for i in range(frames.shape[0]):
                for j in range(frames.shape[1]):
                    frames[i, j] = np.sum(welch(signal[int(i * step):int((i + 1) * step), j],
                                                fs=analysis_obj.sampling_rate)[1])  # total power using Welch's method
        else:
            raise NotImplementedError(f"'{analysis_obj.method}' not implemented.")
        frames_all.append(frames)

    # trim and save frames (remove empty frames)
    frames = frames_all[0]
    for i in range(1, len(frames_all)):
        frames = np.concatenate((frames, frames_all[i]), axis=0)
    frames = frames[np.where(frames.sum(axis=1) != 0), :][0]

    # z-score frame rows/columns
    if 'z_score_rows' in analysis_obj.params.keys() and analysis_obj.params['z_score_rows']:
        frames = scipy.stats.zscore(frames, axis=0)
    if 'z_score_columns' in analysis_obj.params.keys() and analysis_obj.params['z_score_columns']:
        frames = scipy.stats.zscore(frames, axis=1)
    analysis_obj.frames = frames

def compute_correlation_maps(analysis_obj: ArrayAnalysis):
    """
    Compute correlation maps for given array ArrayAnalysis based on the frames
    :param analysis_obj:
    :return:
    """
    if analysis_obj.frames is None:
        compute_frames(analysis_obj)
    frames = analysis_obj.frames

    pairwise_correlations = np.full((analysis_obj.n_channels, analysis_obj.n_channels), -2.)
    correlation_maps = np.full((analysis_obj.n_channels, analysis_obj.array_shape[1], analysis_obj.array_shape[0]), -2.)
    for seed_ch_ix in range(frames.shape[1]):
        # correlate one seed channel with all other channels and create correlation map for each seed channel
        for chn_ix in range(frames.shape[1]):
            corr = pearsonr(frames[:, seed_ch_ix], frames[:, chn_ix])[0]
            if np.isnan(corr):
                corr = 0.
            pairwise_correlations[seed_ch_ix, chn_ix] = corr

            x, y = analysis_obj.get_channel_position(chn_ix)
            correlation_maps[seed_ch_ix, y, x] = corr
    analysis_obj.correlation_maps = correlation_maps

def compute_spontaneous_map(analysis_obj: ArrayAnalysis):
    """

    :param analysis_obj:
    :return:
    """
    if analysis_obj.correlation_maps is None:
        compute_correlation_maps(analysis_obj)

    # load and interpolate correlation maps
    If = analysis_obj.params['interpolation_factor']
    n_channels = analysis_obj.n_channels
    correlation_maps = analysis_obj.correlation_maps
    inter_shape = interpolate_data(correlation_maps[0, :, :], If).shape
    inter_correlation_maps = np.zeros((n_channels, inter_shape[0], inter_shape[1]))
    delete_inds = analysis_obj.get_channels_indices(analysis_obj.deleted_channels)
    for chn_ix in range(n_channels):
        if chn_ix + 1 in analysis_obj.deleted_channels:
            inter_correlation_maps[chn_ix, :, :] = np.full(inter_shape, -2.)
        else:
            inter_correlation_maps[chn_ix, :, :] = interpolate_data(correlation_maps[chn_ix, :, :], If, False, delete_inds)
            good_map_inds = np.where(inter_correlation_maps[chn_ix] >= -1)
    analysis_obj.interpolated_correlation_maps = inter_correlation_maps

    # remove delete channels before PCA fit
    values = []
    fit_values = []
    good_value_inds = [i for i in range(analysis_obj.n_channels) if i + 1 not in analysis_obj.deleted_channels]
    for chn_ix in range(analysis_obj.n_channels):
        if chn_ix + 1 not in analysis_obj.deleted_channels:
            fit_values.append(inter_correlation_maps[chn_ix, good_map_inds[0], good_map_inds[1]])
            values.append(analysis_obj.correlation_values[chn_ix, good_value_inds])
    values = np.array(values)
    fit_values = np.array(fit_values)

    # PCA analysis - compute PCs from interpolated correlation data
    pca = PCA(n_components=10)
    points_pca = pca.fit(fit_values.T).transform(values)
    pca_dims = analysis_obj.params['target_PCA_dims']
    analysis_obj._points_PCA = points_pca
    analysis_obj.compute_new_PC(pca_dims[0],pca_dims[1])

def interpolate_data(values_matrix: np.ndarray, If: int, circular=False, bad_channel_inds=None):
    """

    :param values_matrix:
    :param If:
    :param circular: True if values are circular (0, pi)
    :param bad_channel_inds: indices of channels to be excluded from interpolation
    :return:
    """
    if If == 0:
        if bad_channel_inds is not None:
            for r, c in bad_channel_inds:
                values_matrix[r, c] = -2.
        return values_matrix
    # setup
    n, m = values_matrix.shape
    new_shape = ((2 ** If) * (n + 1) - 1, (2 ** If) * (m + 1) - 1)  # real shape (2**If*n+1, 2**If*m+1)
    interpolated_array = np.zeros(new_shape)
    bad_channel_value = -2.
    if bad_channel_inds is None:
        bad_channel_inds = []
    # remove bad channels
    # for bad_inds in bad_channel_inds:
    #     values_matrix[bad_inds[0], bad_inds[1]] = bad_channel_value
    interpolation_function = np.mean
    if circular:
        interpolation_function = lambda x: scipy.stats.circmean(x, high=np.pi)

    step = 2 ** If - 1  # number of data points between already interpolated data
    x_coords = np.array([x * (step + 1) - 1 for x in range(1, n + 1)])
    y_coords = np.array([y * (step + 1) - 1 for y in range(1, m + 1)])

    # fill real data into interpolated array
    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            interpolated_array[x_coords[i], y_coords[j]] = values_matrix[i, j]
    for i in range(1, If + 1):
        step = int((step - 1) / 2)
        off_array = [-step - 1, step + 1]
        x_coords_new = np.array((x_coords - step - 1).tolist() + [x_coords[-1] + step + 1], dtype=int)
        y_coords_new = np.array((y_coords - step - 1).tolist() + [y_coords[-1] + step + 1], dtype=int)

        # interpolate from data on diagonals
        for x in x_coords_new:
            for y in y_coords_new:
                neighbour_values = []
                for x_off in off_array:
                    for y_off in off_array:
                        if 0 <= x + x_off < new_shape[0] and 0 <= y + y_off < new_shape[1]:
                            if interpolated_array[x + x_off, y + y_off] >= -1:
                                neighbour_values.append(interpolated_array[x + x_off, y + y_off])
                interpolated_array[x, y] = interpolation_function(neighbour_values) \
                    if len(neighbour_values) > 0 else bad_channel_value
                assert not np.isnan(interpolated_array[x, y])
        # and rest of the data
        for x_coords_, y_coords_ in zip([x_coords, x_coords_new], [y_coords_new, y_coords]):
            for x in x_coords_:
                for y in y_coords_:
                    neighbour_values = []
                    for j in range(2):
                        for off in off_array:
                            x_off = j * off
                            y_off = (1 - j) * off
                            if 0 <= x + x_off < new_shape[0] and 0 <= y + y_off < new_shape[1]:
                                if interpolated_array[x + x_off, y + y_off] >= -1:
                                    neighbour_values.append(interpolated_array[x + x_off, y + y_off])
                    interpolated_array[x, y] = interpolation_function(neighbour_values) \
                        if len(neighbour_values) > 0 else bad_channel_value
                    assert not np.isnan(interpolated_array[x, y])
        x_coords = np.array(x_coords.tolist() + x_coords_new.tolist())
        y_coords = np.array(y_coords.tolist() + y_coords_new.tolist())
        x_coords.sort()
        y_coords.sort()

    # cut extra corners
    offset = int(2 ** If / 2 - 1)
    interpolated_array = interpolated_array[offset:new_shape[0] - offset, offset:new_shape[1] - offset]

    # remove bad channels in interpolated array
    step = 2 ** If - 1
    offset = offset + 1
    for bad_inds in bad_channel_inds:
        x = bad_inds[0] * step + bad_inds[0] + offset
        y = bad_inds[1] * step + bad_inds[1] + offset
        interpolated_array[int(x - offset):int(x + offset + 1), int(y - offset):int(y + offset + 1)] = bad_channel_value
    return interpolated_array