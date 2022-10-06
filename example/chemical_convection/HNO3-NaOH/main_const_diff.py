# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import sys
from problem import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([3.0e-3]*self.num_component) #mm^2/sec

problem = Problem(nx=40, ny=144, const_diff=True)
problem.generate_output_instance(sys.argv[1])
problem.define_problem()

problem.setup_flow_solver(r_val=8e6, omega_by_r=1.0)
problem.setup_reaction_solver()
problem.setup_transport_solver()

time_stamps = [3.0, 150.0, 700.0]
problem.solve(dt_val=1e-2, endtime=750.0, time_stamps=time_stamps)
