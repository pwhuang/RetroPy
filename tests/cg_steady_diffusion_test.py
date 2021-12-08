import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '../../')

from reaktoro_transport.problem import TracerTransportProblem
from reaktoro_transport.physics import CGKernel
from reaktoro_transport.solver import SteadyStateSolver

from reaktoro_transport.tests import convergence_rate
from reaktoro_transport.tests.benchmarks import DiffusionBenchmark

from math import isclose

class CGSteadyDiffusionTest(TracerTransportProblem, DiffusionBenchmark,
                            CGKernel, SteadyStateSolver):
    def __init__(self, nx):
        super().__init__(*self.get_mesh_and_markers(nx, 'triangle'))

        self.set_flow_field()
        self.define_problem()
        self.set_problem_bc()

        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='amg')

    def set_problem_bc(self):
        values = DiffusionBenchmark.set_problem_bc(self)
        self.add_component_dirichlet_bc('solute', values=values)

list_of_nx = [10, 20]
element_diameters = []
err_norms = []

for nx in list_of_nx:
    problem = CGSteadyDiffusionTest(nx)
    problem.solve_transport()
    numerical_solution = problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

print(err_norms)

convergence_rate_m = convergence_rate(err_norms, element_diameters)
print(convergence_rate_m)

def test_function():
    assert isclose(convergence_rate_m, 2, rel_tol=0.2)
