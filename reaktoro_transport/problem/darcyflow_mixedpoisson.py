from . import *

class DarcyFlowMixedPoisson(TransportProblemBase, FluidProperty):
    """This class utilizes the mixed Poisson method to solve
    the pressure and velocity of Darcy's flow.
    """

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

        self.__boundary_dict = kwargs

    def set_pressure_ic(self):
        return

    def set_form_and_pressure_bc(self, pressure_bc_val: list):
        """Sets up the FeNiCs form of Darcy flow"""

        if len(pressure_bc_val)!=len(self.__boundary_dict['pressure']):
            raise Exception("length of pressure_bc_val != length of boundary_dict")

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

        mu, k, rho, g = self._mu, self._k, self._rho, self._g
        dx, ds = self.dx, self.ds
        n = self.n

        self.mixed_form = mu/k*inner(v, u)*dx - inner(div(v), p)*dx \
                          - inner(v, rho*g)*dx \
                          + q*div(rho*u)*dx

        for i, marker in enumerate(self.__boundary_dict['pressure']):
            self.mixed_form += pressure_bc_val[i]*inner(n, v)*ds(marker)

    def add_momentum_source(self, sources: list):
        v  = self.__v

        for source in sources:
            self.mixed_form -= inner(v, source)*self.dx

    def set_velocity_bc(self, velocity_bc_val: list):
        """"""

        for i, marker in enumerate(self.__boundary_dict['velocity']):
            self.velocity_bc.append(DirichletBC(self.mixed_func_space.sub(0),
                                                velocity_bc_val[i],
                                                self.boundary_markers, marker))

    def assemble_matrix(self):
        a, self.__L = lhs(self.mixed_form), rhs(self.mixed_form)

        self.__A = assemble(a)
        self.__b = PETScVector()

    def set_solver(self):
        self.__solver = PETScLUSolver()
        prm = self.__solver.parameters

    def solve_flow(self):
        assemble(self.__L, tensor=self.__b)

        for bc in self.velocity_bc:
            bc.apply(self.__A, self.__b)
        self.__solver.solve(self.__A, self.__U1.vector(), self.__b)

        self.velocity_assigner.assign(self.fluid_velocity, self.__U1.sub(0))
        self.pressure_assigner.assign(self.fluid_pressure, self.__U1.sub(1))
