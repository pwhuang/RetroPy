from . import *

class DarcyFlowMixedPoisson(TransportProblemBase):
    """This class utilizes the Augmented Lagrangian Uzawa method to solve
    the pressure and velocity of Darcy's flow.
    """

    def __init__(self, mesh, boundary_markers, domain_markers):
        self.set_mesh(mesh)
        self.set_boundary_markers(boundary_markers)
        self.set_domain_markers(domain_markers)

        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('BDM', 1)

        self.velocity_bc = []

    def set_velocity_fe_space(self, fe_space: str, fe_degree: int):
        self.velocity_finite_element = FiniteElement(fe_space,
                                                     self.mesh.cell_name(),
                                                     fe_degree)

        self.velocity_func_space = FunctionSpace(self.mesh,
                                                 self.velocity_finite_element)

        self.fluid_velocity = Function(self.velocity_func_space)

    def mark_flow_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Keywords
        --------
        velocity: Sets the boundary flow rate.
        pressure: Sets the DirichletBC of pressure.
        """

        self.__boundary_dict = kwargs

    def set_permeability(self, permeability: Expression):
        """Sets the permeability in the unit of length squared."""

        self.__k = interpolate(permeability, self.pressure_func_space)

    def set_porosity(self, porosity: Expression):
        """Sets the porosity in dimensionless unit."""

        self.__phi = interpolate(porosity, self.pressure_func_space)

    def set_fluid_density(self, density: float):
        """Sets the fluid density in the unit of mass over volume."""

        self.__rho = Constant(density)

    def set_fluid_viscosity(self, viscosity: float):
        """Sets fluid dynamic viscosity in the unit of pressure*time."""

        self.__mu = Constant(viscosity)

    def set_gravity(self, gravity: tuple):
        """Sets up the gravity in the body force term of Darcy's law."""

        self.__g = Constant(gravity)

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
        mu, k, rho, g = self.__mu, self.__k, self.__rho, self.__g
        dx, ds = self.dx, self.ds
        n = self.n

        self.mixed_form = mu/k*inner(v, u)*dx(0) - inner(div(v), p)*dx \
                          - inner(v, rho*g)*dx \
                          + q*div(rho*u)*dx

        for i, marker in enumerate(self.__boundary_dict['pressure']):
            self.mixed_form += pressure_bc_val[i]*inner(n, v)*ds(marker)

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
        self.__solver = LUSolver()
        prm = self.__solver.parameters

    def solve_flow(self):
        assemble(self.__L, tensor=self.__b)

        for bc in self.velocity_bc:
            bc.apply(self.__A, self.__b)
        self.__solver.solve(self.__A, self.__U1.vector(), self.__b)

        self.velocity_assigner.assign(self.fluid_velocity, self.__U1.sub(0))
        self.pressure_assigner.assign(self.fluid_pressure, self.__U1.sub(1))
