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

bags = ['bags/step_with_shaping/', 'bags/step_without_shaping/']

topics = {
    '/kinematic_reference/step_power': 'Float64',
    '/impedance_controller/reference': 'KinematicPose',
    '/impedance_controller/status': 'KinematicPose',
}

readers = []
for bag in bags:
  reader = SequentialReader()
  reader.open(StorageOptions(uri=bag, storage_id='mcap'), ConverterOptions())
  readers.append(reader)

# Bag topic types
topics_info = readers[0].get_all_topics_and_types()
bag_types = {t.name: t.type for t in topics_info}
types_filter = {t: get_message(bag_types[t]) for t in topics if t in bag_types}

# Collect
data_frames = []

for reader in readers:
  reader.set_filter(StorageFilter(topics=list(topics.keys())))  # Filter topics
  data = {name: [] for name in {'reference', 'step_power', 'zpower'}}
  t_begin = None
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
  df = df[df['t'] <= 0.249]
  data_frames.append(df)

# Moving average
data_frames[0]['zpower_filt'] = data_frames[0]['zpower']
data_frames[1]['zpower_filt'] = data_frames[1]['zpower'].rolling(window=10).mean() + 446.16

rmse = np.sqrt(
    np.mean(np.square(data_frames[0]['step_power'] - data_frames[0]['zpower_filt']))
)
print(f'With IS (rms): {rmse:.4f} W')

rmse = np.sqrt(
    np.mean(np.square(data_frames[1]['step_power'] - data_frames[1]['zpower_filt']))
)
print(f'Without IS (rms): {rmse:.4f} W')

# Plot
plt.style.use(['science', 'ieee'])

fig, axs = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0)

axs[0].plot(data_frames[0]['t'], data_frames[0]['zpower_filt'],
        label=r'$[{y}^T{u}]_{\Omega}$',
        linestyle='-', linewidth=0.8, color="#188BFF")
axs[0].plot(data_frames[0]['t'], data_frames[0]['step_power'],
        label=r'${P}_{step}$', linestyle='-.', linewidth=0.8, color='k')

axs[1].plot(data_frames[1]['t'], data_frames[1]['zpower_filt'],
        label=r'$[{y}^T{u}]_\Omega$',
        linestyle='-', linewidth=0.8, color="#CA1D0A")
axs[1].plot(data_frames[1]['t'], data_frames[1]['step_power'],
        label=r'${P}_{step}$', linestyle='-.', linewidth=0.8, color='k')

fig.supxlabel('Time (s)', y=0.0)
fig.supylabel('Power (W)')

for ax in axs:
  ax.legend()
  ax.set_ylim(-85, 180)
  ax.grid(True, alpha=0.3)

plt.savefig('figures/step_power_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
