# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *


class StokesFlowBase(FluidProperty):
    """The base class for Stokes flow problems."""

    def __init__(self, marked_mesh):
        self.set_mesh(marked_mesh.mesh)
        self.set_boundary_markers(marked_mesh.boundary_markers)
        self.set_interior_markers(marked_mesh.interior_markers)
        self.set_domain_markers(marked_mesh.domain_markers)
        self.marker_dict = marked_mesh.marker_dict
        self.facet_dict = marked_mesh.facet_dict

    def set_pressure_ic(self, init_cond_pressure):
        """Sets up the initial condition of pressure."""
        self.init_cond_pressure = init_cond_pressure

    def set_pressure_bc(self, bc: dict):
        self.pressure_bc = []

        v, n, ds = self.__v, self.n, self.ds
        mu = self._mu
        u0 = self.fluid_velocity

        for key, pressure_bc in bc.items():
            marker = self.marker_dict[key]
            self.residual_form += pressure_bc * inner(n, v) * ds(marker)
            self.residual_form -= mu * inner(dot(grad(u0), n), v) * ds(marker)

            dofs = locate_dofs_topological(
                V=self.pressure_func_space,
                entity_dim=self.mesh.topology.dim - 1,
                entities=self.facet_dict[key],
            )
            self.pressure_bc.append(dirichletbc(pressure_bc, dofs=dofs))

    def set_velocity_bc(self, bc: dict):
        """
        Arguments
        ---------
        bc
        """

        self.velocity_bc = []
        self.zero_bc = []
        zero_func = Function(self.velocity_func_space)
        zero_func.x.array[:] = 0.0
        zero_func.x.scatter_forward()

        for key, velocity_bc in bc.items():
            dofs = locate_dofs_topological(
                V=self.velocity_func_space,
                entity_dim=self.mesh.topology.dim - 1,
                entities=self.facet_dict[key],
            )
            self.velocity_bc.append(dirichletbc(velocity_bc, dofs=dofs))
            self.zero_bc.append(dirichletbc(zero_func, dofs=dofs))

    def generate_residual_form(self):
        """"""

        V = self.velocity_func_space
        Q = self.pressure_func_space

        self.__v, self.__q = TestFunction(V), TestFunction(Q)

        v = self.__v
        u0, p0 = self.fluid_velocity, self.fluid_pressure

        mu, rho, g = self._mu, self._rho, self._g
        dx = self.dx

        self.residual_form = mu * inner(grad(v), grad(u0)) * dx - inner(div(v), p0) * dx
        self.residual_form -= inner(v, rho * g) * dx

        return self.__v, self.__q

    def add_mass_source_to_residual_form(self, sources: list):
        q = self.__q

        for source in sources:
            self.residual_form -= q * source * self.dx

    def add_momentum_source_to_residual_form(self, sources: list):
        v = self.__v

        for source in sources:
            self.residual_form -= inner(v, source) * self.dx

    def get_flow_residual(self):
        """"""

        residual = assemble_vector(form(self.residual_form))
        residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(residual, bcs=self.zero_bc)
        residual.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        return residual.norm()
