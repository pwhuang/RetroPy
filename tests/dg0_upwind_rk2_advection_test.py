import os
os.environ['OMP_NUM_THREADS'] = '1'

from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import TransientRK2Solver
from reaktoro_transport.manager import XDMFManager

from benchmarks import RotatingCone

from math import isclose

class DG0UpwindRK2AdvectionTest(RotatingCone, DG0Kernel, TransientRK2Solver,
                                XDMFManager):
    def __init__(self, nx, is_output=False):
        super().__init__(*self.get_mesh_and_markers(nx, 'triangle'))

        self.set_flow_field()
        self.define_problem()
        self.set_solver_forms()
        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='amg')

        if is_output==True:
            self.generate_output_instance('rotating_cone_rk2')

    def solve_transport(self, dt_val, timesteps):
        self.dt.assign(dt_val)
        endtime = 0.0

        self.save_to_file(time=endtime)

        for i in range(timesteps):
            self.solve_first_step()
            self.solve_second_step()
            endtime += dt_val
            self.t_end.assign(endtime)
            self.save_to_file(time=endtime)

        self.delete_output_instance()

nx = 30
list_of_dt = [1.5e-2]
timesteps = [70]
err_norms = []

for i, dt in enumerate(list_of_dt):
    problem = DG0UpwindRK2AdvectionTest(nx, is_output=False)
    problem.set_kappa(0.5)
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
allowed_drift_distance = 1e-2

print(mass_error, center_of_mass_error)

def test_function():
    assert mass_error < allowed_mass_error and \
           center_of_mass_error < allowed_drift_distance
