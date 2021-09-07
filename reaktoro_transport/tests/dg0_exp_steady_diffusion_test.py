import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '../../')

from reaktoro_transport.problem import TracerTransportProblemExp
from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import TransientNLSolver

from reaktoro_transport.tests import convergence_rate, quick_plot
from reaktoro_transport.tests.benchmarks import DiffusionBenchmark

from numpy import exp
from dolfin import Constant, Function, norm
from math import isclose

class DG0ExpSteadyDiffusionTest(TracerTransportProblemExp, DiffusionBenchmark,
                                DG0Kernel, TransientNLSolver):
    def __init__(self, nx):
        super().__init__(*self.get_mesh_and_markers(nx, 'triangle'))

        self.set_flow_field()
        self.define_problem()

        self.set_problem_bc()
        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='amg')

    def set_problem_bc(self):
        values = super().set_problem_bc()
        # When solving steady-state problems, the diffusivity of the diffusion
        # boundary is a penalty term to the variational form.
        self.add_component_diffusion_bc('solute', diffusivity=Constant(1e3),
                                        values=values)

    def get_error_norm(self):
        mass_error = Function(self.comp_func_spaces)
        self.fluid_components.vector()[:] = exp(self.fluid_components.vector())

        mass_error.assign(self.fluid_components - self.solution)
        mass_error_norm = norm(mass_error, 'l2')

        return mass_error_norm

list_of_nx = [10]
element_diameters = []
err_norms = []

for i, nx in enumerate(list_of_nx):
    problem = DG0ExpSteadyDiffusionTest(nx)
    problem.solve_transport()
    numerical_solution = problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

print(err_norms)

def test_function():
    assert err_norms[0] < 1e-1
