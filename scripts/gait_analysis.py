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

# from kinematic_pose_msgs.msg import KinematicPose

bag_path = 'bags/gait/'

topics = {
    '/leg_impedance_controller/reference': 'KinematicPose',
    '/leg_impedance_controller/status': 'KinematicPose',
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
series_names = {'foot_x', 'foot_z', 'cmd_power', 'ham_delta'}
data = {name: [] for name in series_names}
time = []

while reader.has_next():
    topic_name, raw, t_ns = reader.read_next()
    if topic_name not in types_filter:
        continue

    if t_begin is None:
        t_begin = t_ns

    t = (t_ns - t_begin) / 1e9  # nanoseconds
    time.append(t)

    msg = deserialize_message(raw, types_filter[topic_name])

    if topic_name == '/leg_impedance_controller/status':
        data['ham_delta'].append(msg.pose_twist.linear.x)
        data['cmd_power'].append(msg.pose_twist.linear.y)
        data['foot_x'].append(msg.pose.position.x)
        data['foot_z'].append(msg.pose.position.z)

df = pd.DataFrame({'t': time})
for series in data:
    df[series] = pd.Series(data[series], index=range(len(data[series])))

df = df[(df['t'] > 1.4) & (df['t'] <= 4.0)]  # Slice

# Integrate the command power
df['cmd_energy'] = (df['cmd_power'] * df['t'].diff().fillna(0)).cumsum()

# Adjust the energy offset:
df['ham_delta'] = df['ham_delta'] - df['ham_delta'].iloc[0]

# Plot
plt.style.use(['science', 'ieee'])

fig, ax = plt.subplots()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy (J)')
ax.plot(df['t'], df['cmd_energy'],
        label=r'$\int \dot{q}^T\tau_{act} dt$',
        linestyle='-', linewidth=0.8, color='blue')
ax.plot(df['t'], df['ham_delta'],
        label=r'$({H}_q - {H}_\Omega)$',
        linestyle='-.', linewidth=0.7, color='k')

ax.set_ylim([-18.0, 6.0])
ax.legend(loc='best', ncols=2)
ax.grid(True, alpha=0.3)

plt.savefig('figures/gait_sim.png', dpi=300, bbox_inches='tight')
plt.show()
