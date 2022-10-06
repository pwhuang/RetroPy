# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from reaktoro_transport.problem import StokesFlowMixedPoisson
from utility_functions import convergence_rate
from benchmarks import StokesFlowBenchmark

from math import isclose
from dolfin import Constant, plot, Expression, PETScOptions, div

import matplotlib.pyplot as plt

class StokesMixedPoissonTest(StokesFlowMixedPoisson, StokesFlowBenchmark):
    """"""

    def __init__(self, nx):
        mesh, boundary_markers, domain_markers = StokesFlowBenchmark.get_mesh_and_markers(self, nx)
        StokesFlowMixedPoisson.__init__(self, mesh, boundary_markers, domain_markers)

        self.set_pressure_fe_space('CG', 1)
        self.set_velocity_vector_fe_space('CG', 2)

        self.set_pressure_ic(Constant(0.0))
        self.get_solution()

        self.set_material_properties()
        self.set_boundary_conditions()
        self.set_momentum_sources()

        self.set_additional_parameters(r_val=0.0)
        self.set_flow_solver_params(solver_type='default', preconditioner='none')

        PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.4)
        PETScOptions.set("pc_hypre_boomeramg_truncfactor", 0.0)
        PETScOptions.set("pc_hypre_boomeramg_print_statistics", 0)
        self.assemble_matrix()

    def set_boundary_conditions(self):
        self.mark_flow_boundary(inlet = [self.marker_dict['left'], self.marker_dict['right']],
                                velocity_bc = [self.marker_dict['top'], self.marker_dict['bottom'],],
                                noslip = [])

        self.generate_form()
        self.set_pressure_dirichlet_bc([Expression(('exp(x[1])*sin(M_PI*x[0])'), degree=1)]*2)
        #self.set_pressure_bc([Expression(('exp(x[1])*sin(M_PI*x[0])'), degree=1)]*2)
        self.set_velocity_bc([Expression(('sin(M_PI*x[1])', 'cos(M_PI*x[0])'), degree=1)]*4)

# nx is the mesh element in one direction.
list_of_nx = [15, 30]
element_diameters = []
p_err_norms = []
v_err_norms = []

for nx in list_of_nx:
    problem = StokesMixedPoissonTest(nx)
    u, p = problem.solve_flow()
    pressure_error_norm, velocity_error_norm = problem.get_error_norm()

    p_err_norms.append(pressure_error_norm)
    v_err_norms.append(velocity_error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

    print(problem.get_flow_residual())

convergence_rate_p = convergence_rate(p_err_norms, element_diameters)
convergence_rate_v = convergence_rate(v_err_norms, element_diameters)

print(convergence_rate_p, convergence_rate_v)

# The relative tolerance of convergence_rate_p is ~1.3 when the number of
# mesh elements is tiny (for faster testings). It converges to 1 when the
# mesh is refined.

def test_function():
    assert isclose(convergence_rate_p, 2, rel_tol=0.05)\
       and isclose(convergence_rate_v, 2, rel_tol=0.05)
