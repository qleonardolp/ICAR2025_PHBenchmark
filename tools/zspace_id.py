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

# Impedance space planar identification (ISPI) use the least square plane
# fitting from "Estimating Surface Normals in Noisy Point Cloud Data".
# [https://dl.acm.org/doi/pdf/10.1145/777792.777840]

import math
import numpy as np


class ZSpaceID:
    """Impedance space planar identification (ISPI)."""

    def __init__(self, window: int):
        if window > 3:
          self.window_size = window
        else:
          self.window_size = 3

        self.buffer = np.zeros((self.window_size, 3))
        self.normal_vector = np.ones(3)
        self.M = np.zeros((3, 3))
        self.counter = 0

    def update(self, e: float, de: float, dde: float):
        if math.isnan(e) or math.isnan(de) or math.isnan(dde):
            return self.normal_vector

        new_point = np.array([e, de, dde])
        # Roll buffer
        for i in range(self.window_size - 1, -1, -1):
          self.buffer[i] = self.buffer[i - 1]
        self.buffer[0] = new_point

        self.counter += 1

        if not (self.counter % self.window_size):
            zspace_mean = self.buffer.mean(0)  # window mean
            for i in range(self.window_size):
                centered_point = self.buffer[i] - zspace_mean
                self.M += np.outer(centered_point, centered_point)
            self.M = self.M / self.window_size
            eigvals, eigvecs = np.linalg.eig(self.M)
            self.normal_vector = eigvecs[-1]  # (?) eigenvector of the least eigenvalue
            # Reset the M matrix
            self.M = np.zeros((3, 3))
            self.counter = 0

    def get_normal(self):
        return self.normal_vector
