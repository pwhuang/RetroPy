import sys
sys.path.insert(0, '../../')

from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import SteadyStateSolver

from reaktoro_transport.tests import convergence_rate
from reaktoro_transport.tests.benchmarks import DiffusionBenchmark

from dolfin import Constant
from math import isclose

class DG0SteadyDiffusionTest(DiffusionBenchmark, DG0Kernel, SteadyStateSolver):
    def __init__(self, nx):
        super().__init__(*self.get_mesh_and_markers(nx, 'quadrilateral'))

        self.set_flow_field()
        self.define_problem()
        self.set_problem_bc()

        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='amg')

    def set_problem_bc(self):
        values = DiffusionBenchmark.set_problem_bc(self)
        # When solving steady-state problems, the diffusivity of the diffusion
        # boundary is a penalty term to the variational form.
        self.add_component_diffusion_bc('solute', diffusivity=Constant(1e3),
                                        values=values)

list_of_nx = [10, 20]
element_diameters = []
err_norms = []

for nx in list_of_nx:
    problem = DG0SteadyDiffusionTest(nx)
    problem.solve_transport()
    numerical_solution = problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

print(err_norms)

convergence_rate_m = convergence_rate(err_norms, element_diameters)
print(convergence_rate_m)

def test_function():
    assert isclose(convergence_rate_m, 1, rel_tol=0.5)
