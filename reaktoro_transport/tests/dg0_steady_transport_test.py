import sys
sys.path.insert(0, '../../')

from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.problem import TracerTransportProblem
from reaktoro_transport.tests import EllipticTransportBenchmark, convergence_rate

from math import isclose

class DG0SteadyTransportTest(EllipticTransportBenchmark, DG0Kernel):
    def __init__(self, nx):
        # TODO: Find out why it does not converge for triangles with 10 < nx <~100.
        TracerTransportProblem.__init__(self, *self.get_mesh_and_markers(nx, 'quadrilateral'))

        self.set_flow_field()
        self.define_problem()
        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='amg')

list_of_nx = [15, 30]
element_diameters = []
err_norms = []

for nx in list_of_nx:
    problem = DG0SteadyTransportTest(nx)
    problem.solve_transport()
    problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

convergence_rate_m = convergence_rate(err_norms, element_diameters)

print(convergence_rate_m)

def test_function():
    assert isclose(convergence_rate_m, 1, rel_tol=0.5)
