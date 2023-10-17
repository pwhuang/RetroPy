# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedRectangleMesh
from retropy.problem import TracerTransportProblem

from dolfinx.fem import Function, Constant, assemble_scalar, form
from mpi4py import MPI


class EllipticTransportBenchmark(TracerTransportProblem):
    """
    This benchmark problem is based on the following work: Optimal order
    convergence of a modified BDM1 mixed finite element scheme for reactive
    transport in porous media by Fabian Brunner et. al., 2012, published in
    Advances in Water Resources. doi: 10.1016/j.advwatres.2011.10.001
    """

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
        self.fluid_velocity.interpolate(lambda x: (0.9 + 0.0 * x[0], 0.9 + 0.0 * x[1]))

    def define_problem(self):
        self.set_components("solute")
        self.set_component_fe_space()
        self.set_advection_velocity()
        self.initialize_form()

        self.set_molecular_diffusivity([1.0])
        self.add_implicit_advection(marker=0)
        self.add_implicit_diffusion("solute", marker=0)

        expr = lambda x: (2.9 - 1.8 * x[0]) * x[1] * (1.0 - x[1]) + (
            2.9 - 1.8 * x[1]
        ) * x[0] * (1.0 - x[0])

        mass_source = Function(self.func_space_list[0])
        mass_source.interpolate(expr)
        self.add_mass_source(["solute"], [mass_source])

        self.mark_component_boundary(**{"solute": self.marker_dict.values()})

        # When solving steady-state problems, the diffusivity of the diffusion
        # boundary is a penalty term to the variational form.
        self.add_component_diffusion_bc(
            "solute",
            diffusivity=Constant(self.mesh, 100.0),
            values=[Constant(self.mesh, 0.0)] * len(self.marker_dict),
        )

    def get_solution(self):
        expr = lambda x: x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])

        self.solution = Function(self.func_space_list[0])
        self.solution.interpolate(expr)

        return self.solution.copy()

    def get_error_norm(self):
        comm = self.mesh.comm
        mass_error = self.fluid_components.sub(0) - self.solution
        mass_error_norm = assemble_scalar(form(mass_error**2 * self.dx))

        return comm.allreduce(mass_error_norm, op=MPI.SUM) ** 0.5
