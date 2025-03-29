"""
Array analysis
This script contains main analysis object

Authors: Karolína Korvasová, Matěj Voldřich
"""

import pickle
from enum import Enum
import neo
import numpy as np
import pandas
import pandas as pd

class METHODS(Enum):
    MUAe = 0
    tMUA = 1
    LFP = 2
    nLFP = 3

class ArrayAnalysis:
    method = None
    params = {}

    array_shape = None
    interpolated_array_shape = None

    segments = None
    analog_signal = None
    sampling_rate = None
    spikes = None
    n_channels = None
    bad_channels = []
    deleted_channels = []
    duration = None
    _locations = {}
    _positions = {}
    _layout = None

    _frames = None
    _correlation_values = None
    _correlation_maps = None
    _interpolated_correlation_values = None
    _interpolated_correlation_maps = None
    _points_PCA = None
    _spontaneous_values = None
    _spontaneous_map = None

    def __init__(self, ID: str, input_type: METHODS, segments: list, layout: pd.DataFrame, params: dict, spacing=400):
        self.id = ID
        self.method = input_type
        self.segments = segments
        self.load_layout(layout, spacing)
        self.params = params

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, frames):
        self._frames = frames

    @property
    def correlation_maps(self):
        if self._correlation_maps is None:
            if self._correlation_values is None:
                return None
            values_matrix = np.zeros((self.n_channels, self.array_shape[0], self.array_shape[1]))
            for chn_ix in range(self.n_channels):
                values_matrix[chn_ix, :, :] = self.values_to_map(self._correlation_values[chn_ix, :])
            return values_matrix
        return self._correlation_maps

    @correlation_maps.setter
    def correlation_maps(self, correlation_maps):
        self._correlation_values = None
        self._correlation_maps = correlation_maps

    @property
    def correlation_values(self):
        if self._correlation_values is None:
            if self._correlation_maps is None:
                return None
            values = np.zeros((self.n_channels, self.n_channels))
            for chn_ix in range(self.n_channels):
                values[chn_ix, :] = self.map_to_values(self._correlation_maps[chn_ix, :, :])
            return values
        return self._correlation_values

    @property
    def interpolated_correlation_values(self):
        if self._interpolated_correlation_values is None:
            values = []
            for chn_ix in range(self.n_channels):
                vals = self._interpolated_correlation_maps[chn_ix, :, :].flatten()
                vals = vals[np.where(vals >= -1)]
                values.append(vals)
            self._interpolated_correlation_values = np.array(values)
        return self._interpolated_correlation_values

    @interpolated_correlation_values.setter
    def interpolated_correlation_values(self, inter_corr_values):
        self._interpolated_correlation_maps = None
        self._interpolated_correlation_values = inter_corr_values

    @property
    def interpolated_correlation_maps(self):
        return self._interpolated_correlation_maps

    @interpolated_correlation_maps.setter
    def interpolated_correlation_maps(self, inter_corr_maps):
        self._interpolated_correlation_values = None
        self._interpolated_correlation_maps = inter_corr_maps

    @property
    def spontaneous_map(self):
        if self._spontaneous_map is None:
            return self.values_to_map(self._spontaneous_values)
        return self._spontaneous_map

    @spontaneous_map.setter
    def spontaneous_map(self, spontaneous_map):
        self._spontaneous_map = spontaneous_map

    @property
    def spontaneous_values(self):
        if self._spontaneous_values is None:
            return self.map_to_values(self._spontaneous_map)
        return self._spontaneous_values

    @spontaneous_values.setter
    def spontaneous_values(self, values):
        for chn in self.bad_channels:
            values[chn - 1] = -2.
        self._spontaneous_values = values

    def compute_new_PC(self,PC1,PC2):
        good_value_inds = [i for i in range(self.n_channels) if i + 1 not in self.deleted_channels]
        points_2d = self._points_PCA[:, [PC1, PC2]]
        self.params["target_PCA_dims"] = [PC1, PC2]
        # PCA analysis - project data onto PCA plane and compute angles
        data_center = np.mean(points_2d, axis=0)
        points_ref = points_2d - data_center
        angles = np.arctan(points_ref[:, 0] / points_ref[:, 1])
        angles[points_ref[:, 1] < 0] += np.pi
        angles -= angles.min()
        labels = angles / 2.
        full_labels = np.full(self.n_channels, -2.)
        full_labels[good_value_inds] = labels
        self._spontaneous_values = full_labels
        self._spontaneous_map = self.values_to_map(self._spontaneous_values)


    def load_layout(self, layout: pandas.DataFrame, spacing=400):
        """

        :param layout: chn, x, y
        :param spacing: spacing between electrode (in um)
        :return:
        """
        layout = trim_layout(layout)
        self._layout = layout
        self.n_channels = len(layout['chn'])
        x_lim = (np.min(layout['x']), np.max(layout['x']))
        x_size = int((x_lim[1] - x_lim[0]) / spacing) + 1
        y_lim = (np.min(layout['y']), np.max(layout['y']))
        y_size = int((y_lim[1] - y_lim[0]) / spacing) + 1
        self.array_shape = (x_size, y_size)

        positions_x = []
        positions_y = []
        for i in range(layout.shape[0]):
            if layout['chn'] is None:
                continue
            positions_x.append((layout['x'][i] - x_lim[0]) / spacing)
            positions_y.append((layout['y'][i] - y_lim[0]) / spacing)
        self.set_positions(layout['chn'], positions_x, positions_y)
        self.set_locations(layout['chn'], layout['x'], layout['y'])

    def set_locations(self, channels, x, y):
        for ix, chn in enumerate(channels):
            self._locations[chn] = (x[ix], y[ix])

    def get_channel_location(self, chn_ix):
        return self._locations[chn_ix + 1]

    def set_positions(self, channels, rows, columns):
        for ix, chn in enumerate(channels):
            self._positions[chn] = (int(rows[ix]), int(columns[ix]))

    def get_channel_position(self, chn_ix):
        return self._positions[chn_ix + 1]

    def get_channels_indices(self, channels):
        """
        Get indices for given channels as (y, x) or (row, col)
        :param channels:
        :return:
        """
        inds = [self._positions[chn] for chn in channels]
        inds = [[ind[1], ind[0]] for ind in inds]
        return inds

    def get_channel_position_interpolated(self, chn_ix):
        raise NotImplementedError()

    def values_to_map(self, values):
        if values is None:
            return None
        values_map = np.full(self.array_shape, -2.)
        for chn in range(self.n_channels):
            x, y = self.get_channel_position(chn)
            values_map[y, x] = values[chn]
        return values_map

    def map_to_values(self, values_matrix):
        if values_matrix is None:
            return None
        values = np.zeros(self.n_channels)
        for chn in range(self.n_channels):
            x, y = self.get_channel_position(chn)
            values[chn] = values_matrix[y, x]
        return values

    def save_lightweight(self, path):
        segments = self.segments
        self.segments = []
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.segments = segments

    def zscore_segments(self):
        from scipy.stats import zscore
        for segment in self.segments:
            ansigs = segment.analogsignals[0]
            segment.analogsignals[0] = neo.AnalogSignal(zscore(ansigs.magnitude, axis=0), units=ansigs.units,
                                                        t_stop=ansigs.t_stop, sampling_rate=ansigs.sampling_rate)

def trim_layout(layout):
    chn, x, y = [], [], []
    for i in range(layout.shape[0]):
        if np.isnan(layout['chn'][i]):
            continue
        chn.append(layout['chn'][i])
        x.append(layout['x'][i])
        y.append(layout['y'][i])
    return pd.DataFrame({'chn': chn, 'x': x, 'y': y})

def load_object(pickle_path):
    with open(pickle_path, "rb") as f:
        arr_obj = pickle.load(f)
    arr_obj.load_layout(arr_obj._layout)
    return arr_obj