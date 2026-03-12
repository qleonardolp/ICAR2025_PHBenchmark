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

import math

from matplotlib import cm

import matplotlib.animation as animation

import matplotlib.pyplot as plt

import numpy as np

# System params
Md = 4.5
Kd = 13.0
Dd = 2 * 0.25 * math.sqrt(Kd * Md)
wn = math.sqrt(Kd/Md)
A = np.array([[0, 1], [-Kd/Md, -Dd/Md]])

# Not working
B = np.array([[0, 0], [1, Kd]])  ## maps f_int and reference (x_d)

# Simulation params
duration = 2 * math.pi / wn  # how many seconds to simulate
duration = 16.0
dt = 0.001
time = np.arange(0.0, duration, dt)
y = np.empty((len(time), 2))  # State time vector

# Initial state
e_0 = 0.254
de_0 = -0.2


def F(t, state, input):
    return A @ state + B @ input


y[0] = np.array([e_0, de_0])
u = np.array([0.0, 0.0])
step_k = 3420
step = 0.23

# Forward Euler integration
for k in range(1, len(time)):
    if k == step_k:
        y[k - 1, 0] = step
    y[k] = y[k - 1] + F(time[k - 1], y[k - 1], u) * dt

hamilton = np.empty((len(time), 1))
hamilton[0] = 0.5 * (de_0*Md*de_0 + e_0*Kd*e_0)

for k in range(1, len(time)):
    dy = y[k] - y[k - 1]
    dde = dy[1] / dt
    dHe = - (Md * dde + Dd * y[k, 1])  # here only the damping is known (structurally)
    dHde = Md * y[k, 1]
    if k == step_k:  ## reset the Hamiltonian when a step is applied
        hamilton[k - 1] = 0.5*Md*y[k, 1]*y[k, 1] + 0.5*Kd*step**2
    hamilton[k] = hamilton[k - 1] + (dHe * dy[0] + dHde * dy[1])

# Plot bounds
x_ub = np.max(y[:, 0])
x_ub = x_ub + abs(x_ub) * 0.10
x_lb = np.min(y[:, 0])
x_lb = x_lb - abs(x_lb) * 0.20
y_ub = np.max(y[:, 1])
y_ub = y_ub + abs(y_ub) * 0.20
y_lb = np.min(y[:, 1])
y_lb = y_lb - abs(y_lb) * 0.10

# Potential gradient field
X = np.arange(x_lb, x_ub, (x_ub - x_lb)/20)
Y = np.arange(y_lb, y_ub, (y_ub - y_lb)/20)
U, V = np.meshgrid(-Kd*X, -Md*Y)


fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(x_lb, x_ub), ylim=(y_lb, y_ub))
ax.set_aspect('equal')

q = ax.quiver(X, Y, U, V, angles='xy', scale_units='width', width=0.0033)

trace, = ax.plot([], [], '--', color='mediumblue', lw=1.3, ms=0.5)
grad_line, = ax.plot([], [], '^-', lw=1.3, ms=0.7)
tang_line, = ax.plot([], [], '^-', lw=1.3, ms=0.7)
acc_line, = ax.plot([], [], '^-', lw=1.3, ms=0.7)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)


def animate(i):
    scale = 0.1
    # Plot a line starting at the
    # phase space point, align with -grad(F)
    gradx = [y[i, 0], (1 - 0.05*Kd) * y[i, 0]]
    grady = [y[i, 1], (1 - 0.05*Md) * y[i, 1]]
    grad_line.set_data(gradx, grady)

    dy = A @ y[i]
    tangx = [y[i, 0], y[i, 0] + scale*dy[0]]
    tangy = [y[i, 1], y[i, 1] + scale*dy[1]]
    tang_line.set_data(tangx, tangy)

    if i == 0:
        ddy = 0 * dy
    else:
        ddy = (A @ y[i] - A @ y[i - 1]) / dt

    accx = [y[i, 0], y[i, 0] + scale*ddy[0]]
    accy = [y[i, 1], y[i, 1] + scale*ddy[1]]
    acc_line.set_data(accx, accy)

    history_x = y[:i, 0]
    history_y = y[:i, 1]
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return trace, grad_line, tang_line, acc_line, time_text


ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)


fig3d = plt.figure(figsize=(5, 4))
ax3d = fig3d.add_subplot(projection='3d')

Xh, Yh = np.meshgrid(X, Y)
H = 0.5*Kd*Xh**2 + 0.5*Md*Yh**2
ax3d.plot_surface(
    Xh, Yh, H, cmap=cm.BuPu, edgecolor='none',
    alpha=0.65, linewidth=0, antialiased=True)

ax3d.plot(y[:, 0].reshape((len(y), 1)), y[:, 1].reshape((len(y), 1)),
          hamilton, color='mediumblue', linewidth=1.6, label='Potential (Traj. Integrated)')

ax3d.legend()
plt.show()
