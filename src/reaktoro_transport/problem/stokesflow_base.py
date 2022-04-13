from . import *

class StokesFlowBase(FluidProperty):
    """The base class for Stokes flow problems."""

    def __init__(self, mesh, boundary_markers, domain_markers):
        self.set_mesh(mesh)
        self.set_boundary_markers(boundary_markers)
        self.set_domain_markers(domain_markers)

    def mark_flow_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Keywords
        --------
        inlet : Sets the boundary flow rate.
        noslip : Sets the boundary to no-slip boundary condition.
        velocity_bc : User defined velocity boundary condition.
        """

        self.stokes_boundary_dict = kwargs

    def set_pressure_ic(self, init_cond_pressure: Expression):
        """Sets up the initial condition of pressure."""
        self.init_cond_pressure = init_cond_pressure

    def set_pressure_bc(self, pressure_bc):
        self.pressure_bc = pressure_bc

    def set_velocity_bc(self, velocity_bc_val: list):
        """
        Arguments
        ---------
        velocity_bc_val : list of Constants,
                          e.g., [Constant((1.0, -1.0)), Constant((0.0, -2.0))]
        """

        if self.mesh.geometric_dimension()==2:
            noslip = Constant((0.0, 0.0))
        elif self.mesh.geometric_dimension()==3:
            noslip = Constant((0.0, 0.0, 0.0))

        self.velocity_bc = []

        for marker in self.stokes_boundary_dict['noslip']:
            self.velocity_bc.append(DirichletBC(self.velocity_func_space,
                                                noslip,
                                                self.boundary_markers, marker))

        for i, marker in enumerate(self.stokes_boundary_dict['velocity_bc']):
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

        mu, rho, g = self._mu, self._rho, self._g

        n = self.n
        dx, ds = self.dx, self.ds

        self.residual_momentum_form = (mu*inner(grad(u0), grad(v)) - inner(p0, div(v)))*dx
        self.residual_momentum_form -= inner(v, rho*g)*dx
        self.residual_mass_form = q*div(rho*u0)*dx

        for i, marker in enumerate(self.stokes_boundary_dict['inlet']):
            self.residual_momentum_form += \
            self.pressure_bc[i]*inner(n, v)*ds(marker) \
            - mu*inner(dot(grad(u0), n), v)*ds(marker)

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

        u0 = self.fluid_velocity

        residual_momentum = assemble(self.residual_momentum_form)
        residual_mass = assemble(self.residual_mass_form)

        for bc in self.velocity_bc:
            bc.apply(residual_momentum, u0.vector())

        residual = residual_momentum.norm('l2')
        residual += residual_mass.norm('l2')

        return residual
