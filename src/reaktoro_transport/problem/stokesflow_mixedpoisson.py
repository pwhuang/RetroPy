from . import *

class StokesFlowMixedPoisson(TransportProblemBase, StokesFlowBase):
    """This class utilizes the mixed Poisson method to solve
    the pressure and velocity of Stokes's flow.
    """

    def __init__(self, mesh, boundary_markers, domain_markers):
        StokesFlowBase.__init__(self, mesh, boundary_markers, domain_markers)

    def generate_form(self):
        """Sets up the FeNiCs form of Stokes flow"""

        self.generate_residual_form()

        W = self.mixed_func_space

        (self.__u, self.__p) = TrialFunctions(W)
        (self.__v, self.__q) = TestFunctions(W)
        self.__U1 = Function(W)

        u, p = self.__u, self.__p
        v, q = self.__v, self.__q

        mu, rho, g = self._mu, self._rho, self._g
        dx, ds = self.dx, self.ds
        n = self.n

        self.__r = Constant(0.0)
        r = self.__r

        self.mixed_form = mu*inner(grad(v), grad(u))*dx - inner(div(v), p)*dx \
                          + r*inner(div(v), div(rho*u))*dx \
                          - inner(v, rho*g)*dx \
                          + q*div(rho*u)*dx

        #self.precond = inner(grad(u), grad(v))*dx + p*q*dx

    def add_mass_source(self, sources):
        q, v, r = self.__q, self.__v, self.__r
        dx = self.dx

        for source in sources:
            self.mixed_form -= q*source*dx + r*inner(div(v), source)*dx

    def add_momentum_source(self, sources: list):
        v  = self.__v

        for source in sources:
            self.mixed_form -= inner(v, source)*self.dx

    def set_pressure_bc(self, pressure_bc):
        super().set_pressure_bc(pressure_bc)

        v = self.__v
        n = self.n
        ds = self.ds
        mu = self._mu
        u = self.__u

        for i, marker in enumerate(self.stokes_boundary_dict['inlet']):
            self.mixed_form += \
            + inner(self.pressure_bc[i]*n, v)*ds(marker) \
            - mu*inner(dot(grad(u), n), v)*ds(marker)

    def set_pressure_dirichlet_bc(self, pressure_bc_val: list):
        self.pressure_dirichlet_bc = []

        for i, marker in enumerate(self.stokes_boundary_dict['inlet']):
            self.pressure_dirichlet_bc.append(DirichletBC(self.mixed_func_space.sub(1),
                                              pressure_bc_val[i],
                                              self.boundary_markers, marker))

    def set_additional_parameters(self, r_val: float, **kwargs):
        self.__r.assign(r_val)

    def assemble_matrix(self):
        a, self.__L = lhs(self.mixed_form), rhs(self.mixed_form)

        self.__A = assemble(a)
        self.__b = PETScVector()

    def set_flow_solver_params(self, solver_type='mumps', preconditioner='ilu'):
        lu_solver_types = ['mumps', 'default', 'superlu', 'superlu_dist']
        iter_solver_types = ['gmres', 'minres', 'bicgstab', 'cg']

        if solver_type not in (lu_solver_types + iter_solver_types):
            raise Exception("Invalid solver type. Possible options: 'mumps'.")

        if solver_type in lu_solver_types:
            self.__solver = PETScLUSolver(solver_type)
        if solver_type in iter_solver_types:
            self.__solver = PETScKrylovSolver(solver_type, preconditioner)

            prm = self.__solver.parameters
            TransportProblemBase.set_default_solver_parameters(prm)

            return prm

    def solve_flow(self, **kwargs):
        assemble(self.__L, tensor=self.__b)

        info('Solving Stokes flow.')

        for bc in self.mixed_vel_bc:
            bc.apply(self.__A, self.__b)

        for bc in self.pressure_dirichlet_bc:
            bc.apply(self.__A, self.__b)

        self.__solver.solve(self.__A, self.__U1.vector(), self.__b)

        self.velocity_assigner.assign(self.fluid_velocity, self.__U1.sub(0))
        self.pressure_assigner.assign(self.fluid_pressure, self.__U1.sub(1))

        return self.fluid_velocity, self.fluid_pressure
