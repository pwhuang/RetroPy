import sys
sys.path.insert(0, '../../')

from reaktoro_transport.physics import DG0Kernel, FluxLimiterCollection
from reaktoro_transport.solver import TransientRK2Solver

from reaktoro_transport.tests.benchmarks import RotatingCone

from dolfin import Constant, assemble, Function
from math import isclose

class DG0FluxLimitAdvectionTest(RotatingCone, DG0Kernel, TransientRK2Solver):
    def __init__(self, nx, is_output=False):
        super().__init__(*self.get_mesh_and_markers(nx, 'triangle'))

        self.set_flow_field()
        self.define_problem()
        self.set_solver_forms()
        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='jacobi')

        if is_output==True:
            self.generate_output_instance('rotating_cone_flux_lim')

    def set_solver_forms(self):
        self.__u_up = Function(self.comp_func_spaces)
        super().set_solver_forms()

        u0, u1 = self.get_solver_functions()
        # Left hand side of the upwind form
        self.L_up1 = self.get_upwind_form(u0)
        self.L_up2 = self.get_upwind_form(u1)

    def add_physics_to_form(self, u, kappa, f_id):
        super().add_physics_to_form(u, kappa, f_id)
        self.add_flux_limiter(u, self.__u_up, k=0.33, kappa=kappa, f_id=f_id)

    def solve_upwind_step(self, L_up):
        self.__u_up.vector()[:] = assemble(L_up).get_local()

    def flux_limiter(self, r):
        return FluxLimiterCollection.vanLeer(r)

    def solve_transport(self, dt_val, timesteps):
        self.dt.assign(dt_val)
        endtime = 0.0

        self.save_to_file(time=endtime)

        for i in range(timesteps):
            self.solve_upwind_step(self.L_up1)
            self.solve_first_step()
            self.solve_upwind_step(self.L_up2)
            self.solve_second_step()
            endtime += dt_val
            self.t_end.assign(endtime)
            self.save_to_file(time=endtime)

        self.delete_output_instance()

nx = 30
list_of_dt = [5e-3]
timesteps = [200]
err_norms = []

for i, dt in enumerate(list_of_dt):
    problem = DG0FluxLimitAdvectionTest(nx, is_output=False)
    problem.set_kappa(1.0)
    initial_mass = problem.get_total_mass()
    initial_center_x, initial_center_y = problem.get_center_of_mass()
    problem.solve_transport(dt_val=dt, timesteps=timesteps[i])

    problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)
    advected_mass = problem.get_total_mass()
    advected_center_x, advected_center_y = problem.get_center_of_mass()

mass_error = abs(initial_mass-advected_mass)
center_of_mass_error = ((advected_center_x - initial_center_x)**2 - \
                        (advected_center_y - initial_center_y)**2)**0.5

allowed_mass_error = 1e-10
allowed_drift_distance = 0.01

print(mass_error, center_of_mass_error)

def test_function():
    assert mass_error < allowed_mass_error and \
           center_of_mass_error < allowed_drift_distance
