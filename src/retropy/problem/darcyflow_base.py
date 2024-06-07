# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *


class DarcyFlowBase(FluidProperty):
    """The base class for Darcy flow problems."""

    def __init__(self, marked_mesh):
        self.set_mesh(marked_mesh.mesh)
        self.set_boundary_markers(marked_mesh.boundary_markers)
        self.set_interior_markers(marked_mesh.interior_markers)
        self.set_domain_markers(marked_mesh.domain_markers)
        self.marker_dict = marked_mesh.marker_dict
        self.facet_dict = marked_mesh.facet_dict

    def set_pressure_ic(self, ic):
        """Sets up the initial condition of pressure."""
        self.fluid_pressure.interpolate(ic)

    def set_pressure_bc(self, bc: dict):
        """Sets up the boundary condition of pressure."""
        self.pressure_bc = bc
        v, n, ds = self.__v, self.n, self.ds

        for key, pressure_bc in self.pressure_bc.items():
            marker = self.marker_dict[key]
            self.residual_momentum_form += pressure_bc * inner(n, v) * ds(marker)

    def add_weak_pressure_bc(self, penalty_value):
        """Sets up the boundary condition of pressure."""
        v, n, ds = self.__v, self.n, self.ds
        alpha = Constant(self.mesh, penalty_value)
        h = Circumradius(self.mesh)
        mu, k = self._mu, self._k
        u, p = self.fluid_velocity, self.fluid_pressure
        mu, k, rho, g = self._mu, self._k, self.fluid_density, self._g

        for key, pressure_bc in self.pressure_bc.items():
            marker = self.marker_dict[key]
            self.residual_momentum_form += alpha * k / mu * ((pressure_bc - p) / h  - rho * dot(g, n) ) * dot(n, v) * ds(marker)
            self.residual_momentum_form += alpha * dot(u, n) * dot(n, v) * ds(marker)

    def set_velocity_bc(self, bc: dict):
        """Sets up the boundary condition of velocity."""

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

        v, q = self.__v, self.__q
        u0, p0 = self.fluid_velocity, self.fluid_pressure

        mu, k, rho, g, phi = self._mu, self._k, self._rho, self._g, self._phi

        dx = self.dx

        self.residual_momentum_form = (
            mu / k * inner(v, u0) * dx - inner(div(v), p0) * dx - inner(v, rho * g) * dx
        )
        self.residual_mass_form = q * div(rho * u0) * dx

    def add_mass_source_to_residual_form(self, sources: list):
        q = self.__q

        for source in sources:
            self.residual_mass_form -= q * source * self.dx

    def add_momentum_source_to_residual_form(self, sources: list):
        v = self.__v

        for source in sources:
            self.residual_momentum_form -= inner(v, source) * self.dx

    def get_flow_residual(self):
        """"""

        residual_momentum = assemble_vector(form(self.residual_momentum_form))
        residual_momentum.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        set_bc(residual_momentum, bcs=self.zero_bc)
        residual_momentum.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        residual_mass = assemble_vector(form(self.residual_mass_form))

        residual = residual_momentum.norm() + residual_mass.norm()

        return residual
