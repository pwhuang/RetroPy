import os
os.environ['OMP_NUM_THREADS'] = '1'

from reaktoro_transport.problem import StokesFlowUzawa
from utility_functions import convergence_rate
from benchmarks import StokesFlowBenchmark

from math import isclose
from dolfin import Constant

class StokesUzawaTest(StokesFlowUzawa, StokesFlowBenchmark):
    """"""

    def __init__(self, nx):
        mesh, boundary_markers, domain_markers = StokesFlowBenchmark.get_mesh_and_markers(self, nx)
        StokesFlowUzawa.__init__(self, mesh, boundary_markers, domain_markers)

        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_vector_fe_space('CR', 1)

        self.set_pressure_ic(Constant(0.0))
        self.get_solution()

        self.set_material_properties()
        self.set_boundary_conditions()
        self.set_momentum_sources()

        self.set_additional_parameters(r_val=3e2, omega_by_r=1.0)
        self.set_flow_solver_params()
        self.assemble_matrix()

# nx is the mesh element in one direction.
list_of_nx = [6, 12]
element_diameters = []
p_err_norms = []
v_err_norms = []

for nx in list_of_nx:
    problem = StokesUzawaTest(nx)
    u, p = problem.solve_flow(target_residual=5e-11, max_steps=20)
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
    assert isclose(convergence_rate_p, 1, rel_tol=0.5)\
       and isclose(convergence_rate_v, 2, rel_tol=0.05)
