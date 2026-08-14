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

from zspace_id import ZSpaceID, FrenetSerret

# System parameters
mi = -0.9
d_desired = mi

# Simulation parameters
dt = 0.002
duration = 16.0
time = np.arange(0.0, duration, dt)
x = np.empty((len(time), 2))  # State time series
ddx = np.zeros((len(time), 1))  # acceleration
ddx_desired = np.zeros((len(time), 1))  # acceleration
f = np.empty(2)

# ZSpace identification
# sys_id = ZSpaceID(10)
sys_id = FrenetSerret(5, dt)
B = np.zeros((len(time), 3))
T = np.zeros((len(time), 3))

damping = np.zeros((len(time), 1))  # damping coefficient (ground truth)
damping_est = np.zeros((len(time), 1))

x[0] = np.array([2.0, 0.0])  # initial state

def F(x: np.ndarray, u: float):
    """Nonlinear dynamics flow for the Van der Pol oscillator."""
    f = np.zeros(2)
    f[0] = x[1]
    f[1] = mi*(1.0 - x[0]*x[0]) * x[1] - x[0] + u
    return f

# Numerical Integration (Fourth order Runge-Kutta)
for k in range(1, len(time)):
    # Desired acceleration
    ddx_desired[k] = -d_desired * x[k - 1, 1] - x[k - 1, 0]
    # Controller
    u = 0.0 * (ddx_desired[k][0] - ddx[k - 1][0])  # scalar
    # Runge-Kutta terms:
    r1 = F(x[k - 1], u)*dt
    r2 = F(x[k - 1] + 0.5*r1, u)*dt
    r3 = F(x[k - 1] + 0.5*r2, u)*dt
    r4 = F(x[k - 1] + r3, u)*dt
    # Runge-Kutta update:
    rk = (r1 + 2*r2 + 2*r3 + r4) * (1/6)
    x[k] = x[k - 1] + rk
    # get acceleration:
    ddx[k] = rk[1] / dt
    # System identification #
    damping[k] = -mi*(1.0 - x[k - 1, 0]*x[k - 1, 0])
    # Compute zspace normal (x, dx, ddx)
    sys_id.update(x[k - 1, 0], x[k - 1, 1], ddx[k][0])
    T[k] = sys_id.get_tangent()
    B[k] = sys_id.get_normal()
    # Damping estimation: (with m = k = 1)
    n1 = sys_id.get_normal()[1]
    d2 = 2*n1*n1/(1 - n1*n1)
    if n1 >= 0:
        damping_est[k] = math.sqrt(d2)
    else:
        damping_est[k] = -math.sqrt(d2)

# Fill the first entry
ddx[0] = ddx[1]
ddx_desired[0] = ddx_desired[1]
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

ax3 = plt.figure().add_subplot(projection='3d')

ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$\dot{x}$')
ax3.set_zlabel(r'$\ddot{x}$')
ax3.plot(x[:, 0], x[:, 1], ddx[:, 0],
    label='Van der Pol', linestyle='-', linewidth=0.7, color='blue')
ax3.plot(x[:, 0], x[:, 1], ddx_desired[:, 0],
    label='Mass-spring-damper', linestyle='-', linewidth=0.7, color='red')
ax3.legend(loc='upper left', columnspacing=0.5)
ax3.grid(True, alpha=0.25)

# Verification
fig3, ax2 = plt.subplots()
ax2.set_xlabel('Time (s)')
ax2.plot(time[5:], damping[5:], label='Damping (ground truth)',
         linestyle='--', linewidth=1.0, color='black')
ax2.plot(time[5:], damping_est[5:], label='Damping (ZSpace)',
         linestyle='-', linewidth=0.9, color='red')
ax2.legend(loc='upper right', columnspacing=0.5)
ax2.grid(True)

# Normal vector
fig4, ax4 = plt.subplots()
ax4.set_xlabel('Time (s)')
ax4.plot(time[5:], B[5:, 0], label='n_0',
         linestyle='-', linewidth=1.0, color='black')
ax4.plot(time[5:], B[5:, 1], label='n_1',
         linestyle='-', linewidth=0.9, color='red')
ax4.plot(time[5:], B[5:, 2], label='n_2',
         linestyle='-', linewidth=0.9, color='blue')
ax4.legend(loc='upper right', columnspacing=0.5)
ax4.grid(True)


# Animation with tangent vector
fig5 = plt.figure()
ax5 = fig5.add_subplot(projection='3d')
ax5.set(xlim3d=(x_lb, x_ub), xlabel=r'$x$')
ax5.set(ylim3d=(y_lb, y_ub), ylabel=r'$\dot{x}$')
ax5.set(zlim3d=(-5.0, 5.0), zlabel=r'$\ddot{x}$')
ax5.set_box_aspect([1, 1, 1])
trace, = ax5.plot([], [], [], '--', color='mediumblue', lw=1.3, ms=0.5)
origin_vec, = ax5.plot([], [], [], '-', color='red', lw=1.3, ms=0.7)
tangent_vec, = ax5.plot([], [], [], '-', color='darkgreen', lw=1.3, ms=0.7)
binormal_vec, = ax5.plot([], [], [], '-', color='blue', lw=1.3, ms=0.7)

def animate5(i):
    trace.set_data_3d(x[:i, 0], x[:i, 1], ddx[:i, 0])

    p = np.concatenate([x[i], ddx[i]])
    t_p = T[i] + p
    b_p = B[i] + p
    origin_vec.set_data_3d([0.0, p[0]], [0.0, p[1]], [0.0, p[2]])
    tangent_vec.set_data_3d([p[0], t_p[0]], [p[1], t_p[1]], [p[2], t_p[2]])
    binormal_vec.set_data_3d([p[0], b_p[0]], [p[1], b_p[1]], [p[2], b_p[2]])
    return trace, origin_vec, tangent_vec, binormal_vec

ani2 = animation.FuncAnimation(
    fig5, animate5, len(x), interval=dt*1000, blit=True)

plt.show()
