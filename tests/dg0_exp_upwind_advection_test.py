# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from reaktoro_transport.problem import TracerTransportProblemExp
from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import TransientNLSolver
from reaktoro_transport.manager import XDMFManager

from benchmarks import RotatingCone

from dolfin import Constant, DOLFIN_EPS, exp, assemble
from math import isclose
from numpy import log, abs

class DG0ExpUpwindAdvectionTest(TracerTransportProblemExp, RotatingCone,
                                DG0Kernel, TransientNLSolver, XDMFManager):
    def __init__(self, nx, is_output=False):
        super().__init__(*self.get_mesh_and_markers(nx, 'triangle'))

        self.set_flow_field()
        self.define_problem()

        if is_output==True:
            self.generate_output_instance('rotating_cone_exp')

    def add_physics_to_form(self, u, kappa=Constant(0.5), f_id=0):
        self.add_explicit_advection(u, kappa, marker=0, f_id=f_id)
        self.add_implicit_advection(kappa, marker=0, f_id=f_id)

    def generate_solver(self):
        super().generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='amg')

    def solve_transport(self, dt_val, timesteps):
        self.dt.assign(dt_val)
        endtime = 0.0

        self.save_to_file(time=endtime)

        for i in range(timesteps):
            self.solve_one_step()
            self.assign_u1_to_u0()
            endtime += dt_val
            self.t_end.assign(endtime)
            self.save_to_file(time=endtime)

        self.delete_output_instance()

    def get_total_mass(self):
        self.total_mass = assemble(exp(self.fluid_components[0])*self.dx)
        return self.total_mass

    def get_center_of_mass(self):
        center_x = assemble(exp(self.fluid_components[0])*self.cell_coord[0]*self.dx)
        center_y = assemble(exp(self.fluid_components[0])*self.cell_coord[1]*self.dx)

        return center_x/self.total_mass, center_y/self.total_mass

nx = 30
list_of_dt = [2e-2]
timesteps = [50]

for i, dt in enumerate(list_of_dt):
    problem = DG0ExpUpwindAdvectionTest(nx, is_output=False)
    problem.generate_solver()

    initial_mass = problem.get_total_mass()
    initial_center_x, initial_center_y = problem.get_center_of_mass()
    problem.solve_transport(dt_val=dt, timesteps=timesteps[i])

    problem.get_solution()
    advected_mass = problem.get_total_mass()
    advected_center_x, advected_center_y = problem.get_center_of_mass()

mass_error = abs(initial_mass-advected_mass)
center_of_mass_error = ((advected_center_x - initial_center_x)**2 - \
                        (advected_center_y - initial_center_y)**2)**0.5

allowed_mass_error = 1e-9
allowed_drift_distance = 0.05

print(mass_error, center_of_mass_error)

def test_function():
    assert mass_error < allowed_mass_error and \
           center_of_mass_error < allowed_drift_distance
