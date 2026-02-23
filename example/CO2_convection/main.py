# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import sys
from problem import Problem


class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity(
            [1.029e-3, 9.311e-3, 5.273e-3, 2.045e-3, 0.804e-3, 1.101e-3]
        )  # mm^2/sec


problem = Problem(nx=240, ny=96, const_diff=False)
problem.generate_output_instance(sys.argv[1])

problem.define_problem()

options_v = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
    "ksp_monitor": None,
}

options_p = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-10,
    "ksp_atol": 1e-12,
    "ksp_max_it": 1000,
    "pc_type": "jacobi",
}

problem.setup_flow_solver(
    r_val=8e4, omega_by_r=1.0, petsc_options_v=options_v, petsc_options_p=options_p
)
problem.setup_reaction_solver()
problem.setup_auxiliary_reaction_solver()
problem.setup_transport_solver()

time_stamps = [780.0]
problem.solve(dt_val=2e-2, endtime=1600.0, time_stamps=time_stamps)
