from . import *

class DarcyFlowBase(FluidProperty):
    """The base class for Darcy flow problems."""

    def __init__(self, mesh, boundary_markers, domain_markers):
        self.set_mesh(mesh)
        self.set_boundary_markers(boundary_markers)
        self.set_domain_markers(domain_markers)

        self.velocity_bc = []

    def mark_flow_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Keywords
        --------
        velocity: Sets the boundary flow rate.
        pressure: Sets the DirichletBC of pressure.
        """

        self.darcyflow_boundary_dict = kwargs

    def set_pressure_ic(self, init_cond_pressure: Expression):
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

        for i, marker in enumerate(self.darcyflow_boundary_dict['velocity']):
            self.velocity_bc.append(DirichletBC(self.velocity_func_space,
                                                velocity_bc_val[i],
                                                self.boundary_markers, marker))

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

        for i, marker in enumerate(self.darcyflow_boundary_dict['pressure']):
            self.residual_momentum_form += self.pressure_bc[i]*inner(n, v) \
                                           *ds(marker)

    def add_momentum_source_to_residual_form(self, sources: list):
        v  = self.__v

        for source in sources:
            self.residual_momentum_form -= inner(v, source)*self.dx

    def get_residual(self):
        """"""

        u0 = self.fluid_velocity

        residual_momentum = assemble(self.residual_momentum_form)
        residual_mass = assemble(self.residual_mass_form)

        for bc in self.velocity_bc:
            bc.apply(residual_momentum, u0.vector())

        residual = residual_momentum.norm('l2')
        residual += residual_mass.norm('l2')

        return residual
