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

    def mark_flow_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Keywords
        --------
        velocity: Sets the boundary flow rate.
        pressure: Sets the DirichletBC of pressure.
        """

        self.darcyflow_boundary_dict = kwargs

    def set_pressure_ic(self, init_cond_pressure):
        """Sets up the initial condition of pressure."""
        self.init_cond_pressure = init_cond_pressure

    def set_pressure_bc(self, pressure_bc: list):
        """Sets up the boundary condition of pressure."""
        self.pressure_bc = pressure_bc

    def set_velocity_bc(self, velocity_bc_val: list):
        """
        Arguments
        ---------
        velocity_bc_val : list of Constants,
                          e.g., [Constant((1.0, -1.0)), Constant((0.0, -2.0))]
        """

        self.velocity_bc = []

        for i, key in enumerate(self.darcyflow_boundary_dict['velocity']):
            dofs = locate_dofs_topological(V = (self.mixed_func_space.sub(0), self.velocity_func_space), 
                                           entity_dim = self.mesh.topology.dim - 1,
                                           entities = self.facet_dict[key])
            bc = dirichletbc(value = velocity_bc_val[i], dofs = dofs,
                             V = self.mixed_func_space.sub(0))
            self.velocity_bc.append(bc)

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

        self.residual_momentum_form = mu/k*inner(v, u0)*dx \
                                      - inner(div(v), p0)*dx \
                                      - inner(v, rho*g)*dx
        self.residual_mass_form = q*div(phi*rho*u0)*dx

        for i, key in enumerate(self.darcyflow_boundary_dict['pressure']):
            marker = self.marker_dict[key]
            self.residual_momentum_form += self.pressure_bc[i] * inner(n, v) * ds(marker)

    def add_mass_source_to_residual_form(self, sources: list):
        q = self.__q

        for source in sources:
            self.residual_mass_form -= q*source*self.dx

    def add_momentum_source_to_residual_form(self, sources: list):
        v  = self.__v

        for source in sources:
            self.residual_momentum_form -= inner(v, source)*self.dx

    def get_flow_residual(self):
        """"""
        residual_momentum = assemble_vector(form(self.residual_momentum_form))
        residual_mass = assemble_vector(form(self.residual_mass_form))

        set_bc(residual_momentum, bcs=self.velocity_bc)

        residual = residual_momentum.norm(2)
        residual += residual_mass.norm(2)

        return residual
