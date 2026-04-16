#!/usr/bin/env python3
# Copyright (c) 2026, qleonardolp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rclpy.serialization import deserialize_message
from rosbag2_py import (
  ConverterOptions,
  SequentialReader,
  StorageFilter,
  StorageOptions
)
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
from scipy.signal import butter, filtfilt

np.set_printoptions(precision=8, suppress=True)

## Utils
# Cartesian axis index map
axis_dict = {'x': 0, 'y': 1, 'z': 2, 'r': 3, 'p': 4, 'w': 5}

def lpfilter(data, cutoff, sampling):
    """Zero phase low-pass filter to smooth timeseries."""
    b, a = butter(4, cutoff, analog=False, fs=sampling)
    y = filtfilt(b, a, data, method='gust')
    return y

## ROS Bag configuration
#bag_path = '../../sys_id/hyl2_PRBS_K640D160M10/'
bag_path = '../../sys_id/hyl2_PRBS_K640D160/'
#bag_path = '../../sys_id/hyl2_PRBS_with_contact/'

controller_name = 'hyl_controller'

topics = {
    f'/{controller_name}/status': 'Float64MultiArray',
    f'/{controller_name}/reference': 'KinematicPose',
}

# ATTENTION: set accordingly {x, y, z, r, p, w}
e_idx = axis_dict['z'] + 6  # Jump the first 6 field
de_idx = e_idx + 6
dde_idx = de_idx + 6

## Bag deserialization and conversion to pandas DataFrame
reader = SequentialReader()
reader.open(
    StorageOptions(uri=bag_path, storage_id='mcap'), ConverterOptions())

# Bag topic types
topics_info = reader.get_all_topics_and_types()
bag_types = {t.name: t.type for t in topics_info}
types_filter = {t: get_message(bag_types[t]) for t in topics if t in bag_types}

# Filter topics
reader.set_filter(StorageFilter(topics=list(topics.keys())))

# Collect
t_begin = None
series_names = {'e', 'de', 'dde', 'm'}
data = {name: [] for name in series_names}
time = []

while reader.has_next():
    topic_name, raw, t_ns = reader.read_next()
    if topic_name not in types_filter:
        continue

    # Align the timestamp based on status topic
    if t_begin is None and topic_name != f'/{controller_name}/status':
        continue

    if t_begin is None:
        t_begin = t_ns

    t = (t_ns - t_begin) / 1e9  # nanoseconds
    time.append(t)

    msg = deserialize_message(raw, types_filter[topic_name])

    # filter Float64MultiArray messages
    if topic_name == list(topics.keys())[0]:
        if np.isfinite(np.array(msg.data)).all():
          data['dde'].append(msg.data[dde_idx])
          data['de'].append(msg.data[de_idx])
          data['e'].append(msg.data[e_idx])
          data['m'].append(msg.data[4])  # m_zz

df = pd.DataFrame({'t': time})
for series in data:
    df[series] = pd.Series(data[series], index=range(len(data[series])))
    df.dropna(how='any', inplace=True)

# Timeseries filtering
fc = 40.0  # cutoff frequency [Hz]
df['e_filt'] = lpfilter(df['e'], fc, 1000)
df['de_filt'] = lpfilter(df['de'], fc, 1000)
df['dde_filt'] = lpfilter(df['dde'], fc, 1000)

# Time slice
df = df[df['t'] > 1.0]


## SVD processing
X = df['e_filt'].to_numpy()
Y = df['de_filt'].to_numpy()
Z = df['dde_filt'].to_numpy()
# Center points
X = X - np.mean(X)
Y = Y - np.mean(Y)
Z = Z - np.mean(Z)

# Reduce dimensionality to
# minimize the computational cost
X = X[::10]
Y = Y[::10]
Z = Z[::10]

# Stack timeseries in a matrix
M = np.array([X, Y, Z])
# Run SVD
U, s, Vh = np.linalg.svd(M.T)
# Get the least singular value eigenvector
plane_n = np.abs(Vh[-1, :])

print(f'SVD-based plane normal: {plane_n}')

# Impedance Params
k_d = 640.0
d_d = 160.0
m_d = 0.223

designed_plane_n = np.array([k_d, d_d, m_d])
designed_norm = np.linalg.norm(designed_plane_n)
designed_plane_n = designed_plane_n / designed_norm
print(f'Designed plane normal:  {designed_plane_n}')

# Equivalent rotation
Rot = Rotation.from_rotvec(np.cross(designed_plane_n, plane_n)).as_matrix()
rotated_params = Rot @ np.array([k_d, d_d, m_d])

np.set_printoptions(precision=6, suppress=True)
print(f'Params from rotation:   {rotated_params}')

sin_theta = np.linalg.norm(np.cross(designed_plane_n, plane_n))
print(f'Rotation angle (°): {np.rad2deg(np.arcsin(sin_theta)):.4f}')
