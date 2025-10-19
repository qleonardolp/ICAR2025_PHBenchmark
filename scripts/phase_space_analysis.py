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

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.animation as animation

# System params
Md = 9.0
Kd = 25.0
Dd = 2 * 0.06 * math.sqrt(Kd * Md)
wn = math.sqrt(Kd/Md)
A = np.array([[0, 1], [-Kd/Md, -Dd/Md]])

# Simulation params
duration = 2 * math.pi / wn  # how many seconds to simulate
# duration = 12.0
dt = 0.002
time = np.arange(0.0, duration, dt)
y = np.empty((len(time), 2))  # State time vector

# Initial state
e_0 = -0.0254
de_0 = 0.0


def F(t, state):
    return A @ state

y[0] = np.array([e_0, de_0])
# Forward Euler integration
for k in range(1, len(time)):
    y[k] = y[k - 1] + F(time[k - 1], y[k - 1]) * dt

# Plot bounds
x_ub = abs(e_0) * 2.0
x_lb = - x_ub
y_ub = 1.5*x_ub
y_lb = -y_ub

# Potential gradient field
X = np.arange(x_lb, x_ub, (x_ub - x_lb)/14)
Y = np.arange(y_lb, y_ub, (y_ub - y_lb)/14)
U, V = np.meshgrid(-Kd*X, -Md*Y)


fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(x_lb, x_ub), ylim=(y_lb, y_ub))
ax.set_aspect('equal')

q = ax.quiver(X, Y, U, V, angles='xy', scale_units='width', width=0.0033)

trace, = ax.plot([], [], '--', lw=1.3, ms=0.5)
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
plt.show()
