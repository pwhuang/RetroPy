# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedRectangleMesh
from retropy.problem import TracerTransportProblemExp

from dolfinx.fem import (VectorFunctionSpace, Function, Constant,
                         assemble_scalar, form)

from mpi4py import MPI
import numpy as np
from ufl import sqrt

class DiffusionBenchmark:
    """"""

    def get_mesh_and_markers(self, nx, mesh_type):
        mesh_factory = MarkedRectangleMesh()
        mesh_factory.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        mesh_factory.set_top_right_coordinates(coord_x = 1.0, coord_y = 1.0)
        mesh_factory.set_number_of_elements(nx, nx)
        mesh_factory.set_mesh_type(mesh_type)

        mesh = mesh_factory.generate_mesh(mesh_shape='crossed')
        boundary_markers, self.marker_dict, self.locator_dict = mesh_factory.generate_boundary_markers()
        interior_markers = mesh_factory.generate_interior_markers()
        domain_markers = mesh_factory.generate_domain_markers()

        self.mesh_characteristic_length = 1.0/nx

        return mesh, boundary_markers, interior_markers, domain_markers

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        self.fluid_velocity = Function(self.Vec_CG1_space)
        self.fluid_velocity.interpolate(lambda x: (0.0*x[0], 0.0*x[0]))

    def define_problem(self):
        self.set_components('solute')
        self.set_component_fe_space()
        self.initialize_form()

        self.set_molecular_diffusivity([1.0])
        self.add_implicit_diffusion('solute', marker=0)

        mass_source = Function(self.CG1_space)
        mass_source.interpolate(lambda x: 2.0*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
        self.add_mass_source(['solute'], [mass_source])

        self.mark_component_boundary(**{'solute': self.marker_dict.values()})

    def add_physics_to_form(self, u0):
        pass

    def add_time_derivatives(self, u0):
        pass

    def set_problem_bc(self):
        """
        This problem requires Dirichlet bc on all boundaries.
        Since the implementation of Dirichlet bcs depends on the solving scheme,
         this method should be defined in tests.
        """

        num_marked_boundaries = len(self.marker_dict)

        uD = Function(self.comp_func_spaces)
        uD.vector.array_w = 0.0
        values = [uD]*num_marked_boundaries

        return values

    def get_solution(self):
        # To match the rank in mixed spaces,
        # one should supply a list of expressions to the Expression Function.
        expr = lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1])

        self.solution = Function(self.func_space_list[0])
        self.solution.interpolate(expr)

        return self.solution

    def get_error_norm(self):
        comm = MPI.COMM_WORLD
        mass_error = Function(self.func_space_list[0])
        mass_error.vector.array_w = self.fluid_components.vector.array_r - self.solution.vector.array_r
        mass_error.x.scatter_forward()

        mass_error_norm = assemble_scalar(form(mass_error**2*self.dx))

        return np.sqrt(comm.allreduce(mass_error_norm, op=MPI.SUM))
