# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedLineMesh
from retropy.problem import TracerTransportProblem

from dolfinx.fem import Function, Constant, assemble_scalar, form
from ufl import exp, as_vector
from mpi4py import MPI


class ParticleAttachment(TracerTransportProblem):
    """
    This benchmark problem is based on the following work: Optimal order
    convergence of a modified BDM1 mixed finite element scheme for reactive
    transport in porous media by Fabian Brunner et. al., 2012, published in
    Advances in Water Resources. doi: 10.1016/j.advwatres.2011.10.001
    """

    @staticmethod
    def solution_expr_c1(t):
        return lambda x: x[0] * (2.0 - x[0]) * x[1] ** 3 / 27.0 * exp(-0.1 * t)

    @staticmethod
    def solution_expr_c2(t):
        return lambda x: (x[0] - 1.0) ** 2 * x[1] ** 2 / 9.0 * exp(-0.1 * t)

    def get_mesh_and_markers(self, nx):
        marked_mesh = MarkedLineMesh(xmin=0.0, xmax=1.0, num_elements=nx)
        self.mesh_characteristic_length = 1.0 / nx

        return marked_mesh

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        self.fluid_velocity = Function(self.Vec_CG1_space)
        self.fluid_velocity.interpolate(lambda x: (0.0 + 0.0 * x[0], -1.0 + 0.0 * x[1]))

    def define_problem(self):
        self.set_components("c1 c2")
        self.set_component_fe_space()
        self.set_advection_velocity()
        self.initialize_form()

        self.set_molecular_diffusivity([0.1, 0.1])

        self.set_component_ics("c1", self.solution_expr_c1(t=0.0))
        self.set_component_ics("c2", self.solution_expr_c2(t=0.0))

        flux_boundaries = [
            self.marker_dict["top"],
            self.marker_dict["left"],
            self.marker_dict["right"],
        ]

        self.mark_component_boundary(
            {
                "c1": flux_boundaries,
                "c2": flux_boundaries,
                "outlet": [self.marker_dict["bottom"]],
            }
        )

    def add_physics_to_form(self, u, kappa=1.0, f_id=0):
        self.add_explicit_advection(u, kappa, marker=0, f_id=f_id)
        self.add_outflow_bc(f_id)

        self.add_implicit_diffusion("c1", kappa, marker=0, f_id=f_id)
        self.add_implicit_diffusion("c2", kappa, marker=0, f_id=f_id)

        source_c1 = Function(self.func_space_list[0])
        source_c1.interpolate(
            lambda x: (
                x[1] * (x[0] - 2.0) * x[0] * (0.1 * x[1] ** 2 + 3.0 * x[1] + 0.6)
                + 0.2 * x[1] ** 3
            )
            / 27
        )

        source_c2 = Function(self.func_space_list[1])
        source_c2.interpolate(
            lambda x: (
                (x[0] - 1.0) ** 2 * (0.1 * x[1] ** 2 + 2.0 * x[1] + 0.2)
                - 0.2 * x[1] ** 2
            )
            / 9
        )

        f_of_t = exp(Constant(self.mesh, -0.1) * self.current_time)

        self.add_mass_source(["c1"], [source_c1 * f_of_t], kappa, f_id)
        self.add_mass_source(["c2"], [source_c2 * f_of_t], kappa, f_id)

        boundary_source_c1 = Function(self.func_space_list[0])
        boundary_source_c1.interpolate(lambda x: x[0] * (2.0 - x[0]))

        boundary_source_c2 = Function(self.func_space_list[1])
        boundary_source_c2.interpolate(lambda x: (x[0] - 1.0) ** 2)

        zero = Constant(self.mesh, 0.0)

        self.add_component_advection_bc(
            "c1", (boundary_source_c1 * f_of_t, zero, zero), kappa, f_id
        )

        self.add_component_advection_bc(
            "c2", (boundary_source_c2 * f_of_t, zero, zero), kappa, f_id
        )

        diff_expr_c1 = [boundary_source_c1 * f_of_t, zero, zero]

        self.add_component_diffusion_bc("c1", self._D[0], diff_expr_c1, kappa, f_id)

        diff_flux_left_right = Function(self.func_space_list[1])
        diff_flux_left_right.interpolate(lambda x: x[1] ** 2 * 2 / 9)

        diff_expr_c2 = [
            (-2.0 / 3) * self._D[1] * boundary_source_c2 * f_of_t,
            -self._D[1] * diff_flux_left_right * f_of_t,
            -self._D[1] * diff_flux_left_right * f_of_t,
        ]

        self.add_component_flux_bc("c2", diff_expr_c2, kappa, f_id)

        self.add_sources(
            as_vector([-u[0] * u[1] * u[1], 2.0 * u[0] * u[1] * u[1]]), kappa, f_id
        )

    def generate_solution(self):
        self.solution = Function(self.comp_func_spaces)
        self.solution.sub(0).interpolate(self.solution_expr_c1(self.current_time.value))
        self.solution.sub(1).interpolate(self.solution_expr_c2(self.current_time.value))

    def get_solution(self):
        return self.solution

    def get_error_norm(self):
        comm = self.mesh.comm
        mass_error = self.fluid_components - self.solution
        mass_error_norm = assemble_scalar(form(mass_error**2 * self.dx))

        return comm.allreduce(mass_error_norm, op=MPI.SUM) ** 0.5
