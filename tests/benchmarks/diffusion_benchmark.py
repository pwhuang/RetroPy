# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedRectangleMesh
from retropy.problem import TracerTransportProblem

from dolfinx.fem import Function, Constant, assemble_scalar, form

from mpi4py import MPI
import numpy as np


class DiffusionBenchmark(TracerTransportProblem):
    """"""

    def get_mesh_and_markers(self, nx, mesh_type):
        marked_mesh = MarkedRectangleMesh()
        marked_mesh.set_bottom_left_coordinates(coord_x=0.0, coord_y=0.0)
        marked_mesh.set_top_right_coordinates(coord_x=1.0, coord_y=1.0)
        marked_mesh.set_number_of_elements(nx, nx)
        marked_mesh.set_mesh_type(mesh_type)
        marked_mesh.locate_and_mark_boundaries()

        marked_mesh.generate_mesh(mesh_shape="crossed")
        marked_mesh.generate_boundary_markers()
        marked_mesh.generate_interior_markers()
        marked_mesh.generate_domain_markers()

        self.mesh_characteristic_length = 1.0 / nx

        return marked_mesh

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        self.fluid_velocity = Function(self.Vec_CG1_space)
        self.fluid_velocity.interpolate(lambda x: (0.0 * x[0], 0.0 * x[0]))

    def define_problem(self):
        self.set_components("solute")
        self.set_component_fe_space()
        self.initialize_form()

        self.mark_component_boundary({"solute": self.marker_dict.values()})

    def add_physics_to_form(self, u, kappa, f_id=0):
        self.set_molecular_diffusivity([1.0])
        self.add_implicit_diffusion("solute", kappa=kappa, marker=0, f_id=f_id)

        mass_source = Function(self.CG1_space)
        mass_source.interpolate(
            lambda x: 2.0 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        )
        self.add_mass_source(["solute"], [mass_source], kappa=kappa)  

    def set_problem_bc(self):
        """
        This problem requires Dirichlet bc on all boundaries.
        Since the implementation of Dirichlet bcs depends on the solving scheme,
         this method should be defined in tests.
        """

        num_marked_boundaries = len(self.marker_dict)
        values = [0.0] * num_marked_boundaries

        return values

    def get_solution(self):
        expr = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

        self.solution = Function(self.func_space_list[0])
        self.solution.interpolate(expr)

        return self.solution

    def get_error_norm(self):
        comm = self.mesh.comm
        mass_error = self.fluid_components.sub(0) - self.solution
        mass_error_norm = assemble_scalar(form(mass_error**2 * self.dx))

        return np.sqrt(comm.allreduce(mass_error_norm, op=MPI.SUM))
