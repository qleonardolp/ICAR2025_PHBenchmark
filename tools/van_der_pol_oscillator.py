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

import math

from matplotlib import cm

import matplotlib.animation as animation

import matplotlib.pyplot as plt

import numpy as np

# System params
mi = 2.5

# Simulation params
dt = 0.001
duration = 16.0
time = np.arange(0.0, duration, dt)
y = np.empty((len(time), 2))  # State time vector
ddx = np.zeros((len(time), 1))  # acceleration
f = np.empty(2)

y[0] = np.array([2.0, 0.0])

# Forward Euler integration (Improve using Runge-Kutta)
for k in range(1, len(time)):
    f[0] = y[k - 1, 1]  # velocity
    x = y[k - 1, 0]
    f[1] = mi*(1 - x*x) * y[k - 1, 1] - x  # acceleration
    ddx[k] = f[1]
    y[k] = y[k - 1] + dt*f

# Plot bounds
x_ub = np.max(y[:, 0])
x_ub = x_ub + abs(x_ub) * 0.10
x_lb = np.min(y[:, 0])
x_lb = x_lb - abs(x_lb) * 0.20
y_ub = np.max(y[:, 1])
y_ub = y_ub + abs(y_ub) * 0.20
y_lb = np.min(y[:, 1])
y_lb = y_lb - abs(y_lb) * 0.10

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(x_lb, x_ub), ylim=(y_lb, y_ub))
ax.set_aspect('equal')

trace, = ax.plot([], [], '--', color='mediumblue', lw=1.3, ms=0.5)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

tail_length = int(1/dt) * 8  # 8 seconds

def animate(i):
    history_x = y[:i, 0]
    history_y = y[:i, 1]
    if i > tail_length:
        history_x = y[i-tail_length:i, 0]
        history_y = y[i-tail_length:i, 1]
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return trace, time_text

ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)

ax3 = plt.figure().add_subplot(projection='3d')

ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$\dot{x}$')
ax3.set_zlabel(r'$\ddot{x}$')
ax3.plot(y[:, 0], y[:, 1], ddx[:, 0], linestyle='-', linewidth=0.7, color='blue')

ax3.legend(loc='upper left', columnspacing=0.5)
ax3.grid(True, alpha=0.25)

plt.show()
