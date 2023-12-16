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

    def set_pressure_ic(self, init_cond_pressure):
        """Sets up the initial condition of pressure."""
        self.init_cond_pressure = init_cond_pressure

    def set_pressure_bc(self, bc: dict):
        """Sets up the boundary condition of pressure."""
        self.pressure_bc = bc

    def set_velocity_bc(self, bc: dict):
        """
        Arguments
        ---------
        velocity_bc_val : list of Constants,
                          e.g., [Constant((1.0, -1.0)), Constant((0.0, -2.0))]
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

        v, q = self.__v, self.__q
        u0, p0 = self.fluid_velocity, self.fluid_pressure

        mu, k, rho, g, phi = self._mu, self._k, self._rho, self._g, self._phi

        n = self.n
        dx, ds, dS = self.dx, self.ds, self.dS

        self.residual_momentum_form = (
            mu / k * inner(v, u0) * dx - inner(div(v), p0) * dx - inner(v, rho * g) * dx
        )
        self.residual_mass_form = q * div(phi * rho * u0) * dx

        for key, pressure_bc in self.pressure_bc.items():
            marker = self.marker_dict[key]
            self.residual_momentum_form += pressure_bc * inner(n, v) * ds(marker)

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
