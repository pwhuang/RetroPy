# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedLineMesh
from retropy.problem import TracerTransportProblem

from dolfinx.fem import Function, functionspace, Constant, assemble_scalar, form

from mpi4py import MPI
import numpy as np


class HeterogeneousDiffusion(TracerTransportProblem):
    """"""

    @staticmethod
    def solution_expr():
        return lambda x: (np.exp(1.0) - np.exp(1.0 - x[0])) / (np.exp(1.0) - 1.0)

    @staticmethod
    def initial_expr():
        return lambda x: 0.0 * x[0]

    def get_mesh_and_markers(self, nx):
        marked_mesh = MarkedLineMesh(xmin=0.0, xmax=1.0, num_elements=nx)
        self.mesh_characteristic_length = 1.0 / nx

        return marked_mesh

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        self.fluid_velocity = Function(self.Vec_CG1_space)
        self.fluid_velocity.x.array[:] = 0.0
        self.fluid_velocity.x.scatter_forward()
        self.set_advection_velocity()

    def define_problem(self):
        self.set_components("C")
        self.set_component_fe_space()
        self.initialize_form()

        D_eff = Function(self.DG0_space)
        D_eff.interpolate(lambda x: np.exp(x[0]))
        self.set_molecular_diffusivity([D_eff])

        self.mark_component_boundary(
            {"C": [self.marker_dict["left"], self.marker_dict["right"]]}
        )

        self.set_component_ics("C", self.initial_expr())

    def add_physics_to_form(self, u, **kwargs):
        one = Constant(self.mesh, 1.0)
        
        self.add_implicit_diffusion("C", kappa=one, marker=0)
        self.add_component_diffusion_bc(
            "C",
            diffusivity=Constant(self.mesh, 1e2),
            values=[0.0, 1.0],
        )

    def get_solution(self):
        self.interpolation_space = functionspace(self.mesh, ('P', 2))
        expr = self.solution_expr()

        self.u_solution = Function(self.interpolation_space)
        self.u_solution.interpolate(expr)

        self.solution = Function(self.comp_func_spaces)
        self.solution.sub(0).interpolate(self.u_solution)

        return self.solution

    def get_error_norm(self):
        u_numerical = Function(self.interpolation_space)
        u_numerical.interpolate(self.fluid_components.sub(0))

        comm = self.mesh.comm
        mass_error = u_numerical - self.u_solution
        mass_error_norm = assemble_scalar(form(mass_error**2 * self.dx))

        return np.sqrt(comm.allreduce(mass_error_norm, op=MPI.SUM))
