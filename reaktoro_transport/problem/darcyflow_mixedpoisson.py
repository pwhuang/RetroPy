from . import *

class DarcyFlowMixedPoisson(TransportProblemBase, DarcyFlowBase):
    """This class utilizes the mixed Poisson method to solve
    the pressure and velocity of Darcy's flow.
    """

    def __init__(self, mesh, boundary_markers, domain_markers):
        DarcyFlowBase.__init__(self, mesh, boundary_markers, domain_markers)

    def generate_form(self):
        """Sets up the FeNiCs form of Darcy flow"""

        if len(self.pressure_bc)!=len(self.darcyflow_boundary_dict['pressure']):
            raise Exception("length of pressure_bc != length of boundary_dict")

        self.func_space_list = [self.velocity_finite_element,
                                self.pressure_finite_element]

        self.mixed_func_space = FunctionSpace(self.mesh,
                                              MixedElement(self.func_space_list))

        self.velocity_assigner = FunctionAssigner(self.velocity_func_space,
                                                  self.mixed_func_space.sub(0))

        self.pressure_assigner = FunctionAssigner(self.pressure_func_space,
                                                  self.mixed_func_space.sub(1))

        W = self.mixed_func_space

        (self.__u, self.__p) = TrialFunctions(W)
        (self.__v, self.__q) = TestFunctions(W)
        self.__U1 = Function(W)

        u, p = self.__u, self.__p
        v, q = self.__v, self.__q

        mu, k, rho, g, phi = self._mu, self._k, self._rho, self._g, self._phi
        dx, ds = self.dx, self.ds
        n = self.n

        self.__r = Constant(0.0)
        r = self.__r

        self.mixed_form = mu/k*inner(v, u)*dx - inner(div(v), p)*dx \
                          + r*inner(div(v), div(phi*rho*u))*dx \
                          - inner(v, rho*g)*dx \
                          + q*div(phi*rho*u)*dx

        for i, marker in enumerate(self.darcyflow_boundary_dict['pressure']):
            self.mixed_form += self.pressure_bc[i]*inner(n, v)*ds(marker)

        self.functions_to_save = [self.fluid_pressure, self.fluid_velocity]

    def add_mass_source(self, sources):
        q, v, r = self.__q, self.__v, self.__r
        dx = self.dx

        for source in sources:
            self.mixed_form -= q*source*dx + r*inner(div(v), source)*dx

    def add_momentum_source(self, sources: list):
        v  = self.__v

        for source in sources:
            self.mixed_form -= inner(v, source)*self.dx

    def set_velocity_bc(self, velocity_bc_val: list):
        """"""

        DarcyFlowBase.set_velocity_bc(self, velocity_bc_val)
        self.mixed_velocity_bc = []
        markers = self.darcyflow_boundary_dict['velocity']

        for i, marker in enumerate(markers):
            self.mixed_velocity_bc.append(DirichletBC(self.mixed_func_space.sub(0),
                                                velocity_bc_val[i],
                                                self.boundary_markers, marker))

    def set_additional_parameters(self, r_val: float, **kwargs):
        self.__r.assign(r_val)

    def assemble_matrix(self):
        a, self.__L = lhs(self.mixed_form), rhs(self.mixed_form)

        self.__A = assemble(a)
        self.__b = PETScVector()

    def set_solver(self, solver_type='mumps', preconditioner='ilu'):
        lu_solver_types = ['mumps', 'default', 'superlu', 'petsc']
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

        for bc in self.mixed_velocity_bc:
            bc.apply(self.__A, self.__b)
        self.__solver.solve(self.__A, self.__U1.vector(), self.__b)

        self.velocity_assigner.assign(self.fluid_velocity, self.__U1.sub(0))
        self.pressure_assigner.assign(self.fluid_pressure, self.__U1.sub(1))
