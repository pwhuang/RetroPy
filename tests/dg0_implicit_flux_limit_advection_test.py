# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from retropy.physics import DG0Kernel, FluxLimiterCollection
from retropy.solver import TransientSolver
from retropy.manager import XDMFManager

from benchmarks import RotatingCone

from dolfin import Constant, assemble, Function
from math import isclose
from ufl import tanh

class DG0ImplicitFluxLimitAdvectionTest(RotatingCone, DG0Kernel,
                                        TransientSolver, XDMFManager):
    def __init__(self, nx, is_output=False):
        super().__init__(*self.get_mesh_and_markers(nx, 'triangle'))

        self.set_flow_field()
        self.define_problem()
        self.set_solver_forms()
        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='jacobi')

        if is_output==True:
            self.generate_output_instance('rotating_cone_implicit_flux_lim')

    def set_solver_forms(self):
        self.__u_up = Function(self.comp_func_spaces)
        u0 = self.fluid_components
        self.L_up1 = self.get_upwind_form(u0)

    def add_physics_to_form(self, u, kappa=Constant(1.0), f_id=0):
        super().add_physics_to_form(u, kappa, f_id)
        self.add_implicit_flux_limiter(u, self.__u_up, kappa=kappa, f_id=f_id)

    def solve_upwind_step(self, L_up):
        self.__u_up.vector()[:] = assemble(L_up).get_local()

    def flux_limiter(self, r):
        return 0.5*(tanh(1e4*(r-0.25))+1)

    def solve_transport(self, dt_val, timesteps):
        self.dt.assign(dt_val)
        endtime = 0.0

        self.save_to_file(time=endtime)

        for i in range(timesteps):
            self.solve_upwind_step(self.L_up1)
            self.solve_one_step()
            endtime += dt_val
            self.t_end.assign(endtime)
            self.save_to_file(time=endtime)

        self.delete_output_instance()

nx = 30
list_of_dt = [5e-3]
timesteps = [200]
err_norms = []

for i, dt in enumerate(list_of_dt):
    problem = DG0ImplicitFluxLimitAdvectionTest(nx, is_output=False)

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
allowed_drift_distance = 0.02

print(mass_error, center_of_mass_error)

def test_function():
    assert mass_error < allowed_mass_error and \
           center_of_mass_error < allowed_drift_distance
