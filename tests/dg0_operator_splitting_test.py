# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from dg0_particle_reversible_attachment_test import DG0ParticleReversibleAttachmentTest

from utility_functions import convergence_rate

from math import isclose
import numpy as np


class DG0OperatorSplittingTest(DG0ParticleReversibleAttachmentTest):
    def langmuir_kinetics(self, C, S):
        return 0.0

    def solve_reaction(self):
        C = self.get_solver_u1().x.array.reshape(-1, self.num_component)[:, 0]
        S = self.fluid_components.x.array.reshape(-1, self.num_component)[:, 1]

        Smax = self._M.value
        ka = self.Da_att.value / Smax
        kd = self.Da_det.value
        CT = C + Smax * S

        S = self.solve_langmuir_kinetics(CT, S, Smax, ka, kd, self.dt.value)
        C = CT - Smax * S

        self.get_solver_u1().x.array[:] = np.array([C, S]).T.flatten()

    @staticmethod
    def solve_langmuir_kinetics(CT, S0, Smax, ka, kd, dt):
        B = ka * (CT + Smax) + kd
        k1 = np.sqrt(B**2 - 4.0 * ka**2 * Smax * CT)
        k2 = ka * Smax
        Se = 0.5 * (B - k1) / (ka * Smax)

        return Se + k1 * (S0 - Se) / (
            np.exp(k1 * dt) * (k1 - k2 * (S0 - Se)) + k2 * (S0 - Se)
        )

    def solve_one_step(self):
        super().solve_one_step()
        self.solve_reaction()


if __name__ == "__main__":
    Pe, Da_att, Da_det, M, t0 = np.inf, 5.5, 1.3, 1.6, 1.0

    nx_list = [33, 66]
    dt_list = [2e-2, 1e-2]
    timesteps = [50, 100]
    err_norms = []

    for nx, dt, timestep in zip(nx_list, dt_list, timesteps):
        problem = DG0OperatorSplittingTest(nx, Pe, Da_att, Da_det, M, t0)
        problem.solve_transport(dt_val=dt, timesteps=timestep)
        problem.inlet_flux.value = 0.0
        problem.solve_transport(dt_val=dt, timesteps=int(0.5 * timestep))

        problem.generate_solution()
        error_norm = problem.get_error_norm()
        err_norms.append(error_norm)

        # problem.mpl_output()

    print(err_norms)

    convergence_rate_m = convergence_rate(err_norms, dt_list)
    print(convergence_rate_m)

    def test_function():
        assert isclose(convergence_rate_m[0], 1, rel_tol=0.2)
