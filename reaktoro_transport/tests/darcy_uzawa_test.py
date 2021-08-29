import sys
sys.path.insert(0, '../../')

from reaktoro_transport.problem import DarcyFlowUzawa
from reaktoro_transport.tests import convergence_rate
from reaktoro_transport.tests.benchmarks import DarcyFlowBenchmark

from math import isclose
from dolfin import Constant

class DarcyUzawaTest(DarcyFlowUzawa, DarcyFlowBenchmark):
    """"""

    def __init__(self, nx):
        mesh, boundary_markers, domain_markers = DarcyFlowBenchmark.get_mesh_and_markers(self, nx)
        DarcyFlowUzawa.__init__(self, mesh, boundary_markers, domain_markers)

        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('BDM', 1)

        self.set_pressure_ic(Constant(0.0))
        self.get_solution()

        DarcyFlowBenchmark.set_material_properties(self)
        DarcyFlowBenchmark.set_boundary_conditions(self)
        DarcyFlowBenchmark.set_momentum_sources(self)

        self.set_additional_parameters(r_val=1.0, omega_by_r=1.2)
        self.set_solver()
        self.assemble_matrix()

# nx is the mesh element in one direction.
list_of_nx = [10, 20]
element_diameters = []
p_err_norms = []
v_err_norms = []

for nx in list_of_nx:
    problem = DarcyUzawaTest(nx)
    problem.solve_flow(target_residual=1e-10, max_steps=50)
    pressure_error_norm, velocity_error_norm = problem.get_error_norm()

    p_err_norms.append(pressure_error_norm)
    v_err_norms.append(velocity_error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

    print(problem.get_residual())

convergence_rate_p = convergence_rate(p_err_norms, element_diameters)
convergence_rate_v = convergence_rate(v_err_norms, element_diameters)

print(convergence_rate_p, convergence_rate_v)

def test_function():
    assert isclose(convergence_rate_p, 2, rel_tol=0.05)\
       and isclose(convergence_rate_v, 2, rel_tol=0.05)
