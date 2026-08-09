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

from zspace_id import ZSpaceID

# System parameters
mi = -0.88

# Simulation parameters
dt = 0.002
duration = 16.0
time = np.arange(0.0, duration, dt)
x = np.empty((len(time), 2))  # State time series
ddx = np.zeros((len(time), 1))  # acceleration
f = np.empty(2)

# ZSpace identification
sys_id = ZSpaceID(20)
N = np.zeros((len(time), 3))

damping = np.zeros((len(time), 1))  # damping coefficient (ground truth)
damping_est = np.zeros((len(time), 1))

x[0] = np.array([2.0, 0.0])  # initial state

def F(x):
    """Nonlinear dynamics flow for the Van der Pol oscillator."""
    f = np.zeros(2)
    f[0] = x[1]
    f[1] = mi*(1.0 - x[0]*x[0]) * x[1] - x[0]
    return f

# Numerical Integration (Fourth order Runge-Kutta)
for k in range(1, len(time)):
    # Runge-Kutta terms:
    r1 = F(x[k - 1])*dt
    r2 = F(x[k - 1] + 0.5*r1)*dt
    r3 = F(x[k - 1] + 0.5*r2)*dt
    r4 = F(x[k - 1] + r3)*dt
    # Runge-Kutta update:
    rk = (r1 + 2*r2 + 2*r3 + r4) * (1/6)
    x[k] = x[k - 1] + rk
    # get acceleration:
    ddx[k] = rk[1] / dt
    # System identification #
    damping[k] = -mi*(1.0 - x[k - 1, 0]*x[k - 1, 0])
    # Compute zspace normal (x, dx, ddx)
    sys_id.update(x[k - 1, 0], x[k - 1, 1], ddx[k][0])
    N[k] = sys_id.get_normal()
    # Damping estimation: (with k = 1)
    damping_est[k] = sys_id.get_normal()[1] / sys_id.get_normal()[0]

# Fill the first entry
ddx[0] = ddx[1]
damping[0] = damping[1]
damping_est[0] = damping_est[1]

# Plot bounds
x_ub = np.max(x[:, 0])
x_ub = x_ub + abs(x_ub) * 0.10
x_lb = np.min(x[:, 0])
x_lb = x_lb - abs(x_lb) * 0.20
y_ub = np.max(x[:, 1])
y_ub = y_ub + abs(y_ub) * 0.20
y_lb = np.min(x[:, 1])
y_lb = y_lb - abs(y_lb) * 0.10

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(x_lb, x_ub), ylim=(y_lb, y_ub))
ax.set_aspect('equal')

trace, = ax.plot([], [], '--', color='mediumblue', lw=1.3, ms=0.5)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

tail_length = int(1/dt) * 8  # 8 seconds

def animate(i):
    history_x = x[:i, 0]
    history_y = x[:i, 1]
    if i > tail_length:
        history_x = x[i-tail_length:i, 0]
        history_y = x[i-tail_length:i, 1]
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return trace, time_text

ani = animation.FuncAnimation(
    fig, animate, len(x), interval=dt*1000, blit=True)

ax3 = plt.figure().add_subplot(projection='3d')

ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$\dot{x}$')
ax3.set_zlabel(r'$\ddot{x}$')
ax3.plot(x[:, 0], x[:, 1], ddx[:, 0], linestyle='-', linewidth=0.7, color='blue')
ax3.legend(loc='upper left', columnspacing=0.5)
ax3.grid(True, alpha=0.25)

# Verification
fig3, ax2 = plt.subplots()
ax2.set_xlabel('Time (s)')
ax2.plot(time, damping, label='Damping (ground truth)',
         linestyle='--', linewidth=1.0, color='black')
ax2.plot(time, damping_est, label='Damping (ZSpace)',
         linestyle='-', linewidth=0.9, color='red')
ax2.legend(loc='upper right', columnspacing=0.5)
ax2.grid(True)

# Normal vector
fig4, ax4 = plt.subplots()
ax4.set_xlabel('Time (s)')
ax4.plot(time, N[:, 0], label='n_0',
         linestyle='-', linewidth=1.0, color='black')
ax4.plot(time, N[:, 1], label='n_1',
         linestyle='-', linewidth=0.9, color='red')
ax4.plot(time, N[:, 2], label='n_2',
         linestyle='-', linewidth=0.9, color='blue')
ax4.legend(loc='upper right', columnspacing=0.5)
ax4.grid(True)

plt.show()
