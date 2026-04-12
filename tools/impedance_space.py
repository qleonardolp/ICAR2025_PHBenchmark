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
import scienceplots  # noqa: F401
from scipy.signal import butter, filtfilt

# Cartesian axis index map
axis_dict = {'x': 0, 'y': 1, 'z': 2, 'r': 3, 'p': 4, 'w': 5}

def lpfilter(data, cutoff, sampling):
    """Zero phase low-pass filter to smooth timeseries."""
    b, a = butter(4, cutoff, analog=False, fs=sampling)
    y = filtfilt(b, a, data, method='gust')
    return y

# bag_path = '../../bags/rosbag2_2026_03_11-19_25_59/'
bag_path = '../../sys_id/hyl_PRBS_K640D160/'
controller_name = 'hyl_controller'

# ATTENTION: set accordingly {x, y, z, r, p, w}
e_idx = axis_dict['z'] + 6  # Jump the first 6 field
de_idx = e_idx + 6
dde_idx = de_idx + 6

topics = {
    f'/{controller_name}/status': 'Float64MultiArray',
    f'/{controller_name}/reference': 'KinematicPose',
}

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
filt = True
if filt:
    fc = 40.0  # cutoff frequency [Hz]
    df['e_filt'] = lpfilter(df['e'], fc, 1000)
    df['de_filt'] = lpfilter(df['de'], fc, 1000)
    df['dde_filt'] = lpfilter(df['dde'], fc, 1000)
else:
    df['e_filt'] = df['e']
    df['de_filt'] = df['de']
    df['dde_filt'] = df['dde']

# Time slice
df = df[df['t'] > 1.0]

# Plot
# plt.style.use(['science', 'ieee'])

fig, ax = plt.subplots()
ax.set_xlabel('Time (s)')
ax.plot(df['t'], df['m'], label=r'$m_{zz}$', linestyle='-', linewidth=0.8, color='blue')
# ax.plot(df['t'], df['de_filt'], label=r'$\ddot{e}$', linewidth=0.8, color='k')
ax.grid(True)

ax3 = plt.figure().add_subplot(projection='3d')

ax3.set_xlabel(r'$e$')
ax3.set_ylabel(r'$\dot{e}$')
ax3.set_zlabel(r'$\ddot{e}$')
ax3.plot(df['e_filt'], df['de_filt'], df['dde_filt'],
         linestyle='-', linewidth=0.7, color='blue')

ax3.legend(loc='upper left', columnspacing=0.5)
ax3.grid(True, alpha=0.25)

plt.show()
