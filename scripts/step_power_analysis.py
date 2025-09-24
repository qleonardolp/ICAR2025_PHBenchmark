#!/usr/bin/env python3
# Copyright (c) 2025, qleonardolp
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

import matplotlib.pyplot as plt

import numpy as np

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

# bag_path = 'bags/step_with_shaping/'
bag_path = 'bags/step_without_shaping/'

topics = {
    '/kinematic_reference/step_power': 'Float64',
    '/impedance_controller/reference': 'KinematicPose',
    '/impedance_controller/status': 'KinematicPose',
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
series_names = {'reference', 'step_power', 'zpower'}
data = {name: [] for name in series_names}
time = []

while reader.has_next():
    topic_name, raw, t_ns = reader.read_next()
    if topic_name not in types_filter:
        continue

    msg = deserialize_message(raw, types_filter[topic_name])

    if t_begin is None and topic_name != '/kinematic_reference/step_power':
        continue

    if t_begin is None:
        t_begin = t_ns

    t = (t_ns - t_begin) / 1e9  # nanoseconds
    time.append(t)

    if topic_name == '/kinematic_reference/step_power':
        data['step_power'].append(msg.data)

    if topic_name == '/impedance_controller/reference':
        data['reference'].append(msg.pose.position.x)

    if topic_name == '/impedance_controller/status':
        data['zpower'].append(msg.pose_twist.angular.z)


df = pd.DataFrame({'t': time})
for series in data:
    df[series] = pd.Series(data[series], index=range(len(data[series])))

df = df[df['t'] <= 0.249]  # Slice

# Moving average
# df['zpower_filt'] = df['zpower'].ewm(alpha=0.05, adjust=False).mean()
df['zpower_filt'] = df['zpower'].rolling(window=10).mean() + 446.16

rmse = np.sqrt(
    np.mean(np.square(df['step_power'] - df['zpower_filt']))
)

print(f'Fidelity rms: {rmse}')
# With IS (zpower): 13.11383 W
# Without IS (filt): 18.01598 W

# Plot
plt.style.use(['science', 'ieee'])

fig, ax = plt.subplots()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Power (W)')
ax.plot(df['t'], df['zpower_filt'],
        label=r'$[{y}^T{u}]_\Omega$',
        linestyle='-', linewidth=0.8, color='blue')
ax.plot(df['t'], df['step_power'],
        label=r'${P}_{step}$', linestyle='-.', linewidth=0.8, color='k')
ax.legend()
ax.grid(True, alpha=0.3)

# plt.savefig('figures/step_power_with_shaping.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/step_power_without_shaping.png', dpi=300, bbox_inches='tight')
plt.show()
