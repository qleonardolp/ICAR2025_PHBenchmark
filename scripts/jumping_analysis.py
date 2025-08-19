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

bag_path = 'bags/jumping/'

topics = {
    '/joint_states': 'JointState',
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
series_names = {'ref', 'foot_z', 'support_joint', 'cmd_power', 'ham_delta'}
data = {name: [] for name in series_names}
time = []

while reader.has_next():
    topic_name, raw, t_ns = reader.read_next()
    if topic_name not in types_filter:
        continue

    # Align the timestamp based on reference topic
    if t_begin is None and topic_name != '/leg_impedance_controller/reference':
        continue

    if t_begin is None:
        t_begin = t_ns

    t = (t_ns - t_begin) / 1e9  # nanoseconds
    time.append(t)

    msg = deserialize_message(raw, types_filter[topic_name])

    if topic_name == '/joint_states':
        data['support_joint'].append(msg.position[3])

    if topic_name == '/leg_impedance_controller/reference':
        data['ref'].append(msg.pose.position.z)

    if topic_name == '/leg_impedance_controller/status':
        data['ham_delta'].append(msg.pose_twist.linear.x)
        data['cmd_power'].append(msg.pose_twist.linear.y)
        data['foot_z'].append(msg.pose.position.z)

df = pd.DataFrame({'t': time})
for series in data:
    df[series] = pd.Series(data[series], index=range(len(data[series])))

df = df[(df['t'] > 0.5) & (df['t'] <= 7.0)]  # Slice

# Integrate the command power
df['cmd_energy'] = (df['cmd_power'] * df['t'].diff().fillna(0)).cumsum()

# Adjust the energy offset:
df['ham_delta'] = df['ham_delta'] - df['ham_delta'].iloc[0]

# Energy peak before the jump -536.621 J
energy_peak = df['ham_delta'].min()

print(f'Energy peak: {energy_peak} J')

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
        linewidth=0.8, color='k')

ax.set_ylim([-300.0, 300.0])
ax.legend(loc='upper left', columnspacing=0.5)
ax.grid(True, alpha=0.3)
plt.savefig('figures/jumping_sim.png', dpi=300)


fig2, ax2 = plt.subplots()
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Vertical position (m)')
ax2.plot(df['t'], df['support_joint'] + 0.4,
         label='Trunk absolute height',
         linestyle='-', linewidth=0.8, color='orange')
ax2.plot(df['t'], df['ref'], label=r'$z_d$', linestyle='-.', linewidth=0.6, color='k')
ax2.plot(df['t'], df['foot_z'], label=r'$z$', linestyle='-', linewidth=0.7, color='blue')

ax2.set_ylim([-1.6, 2.0])
ax2.legend(loc='upper right', ncols=3)
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
plt.savefig('figures/jumping_z_sim.png', dpi=300)
plt.show()
