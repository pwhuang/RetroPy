from . import *

class DarcyFlowAngot(TransportProblemBase, FluidProperty):
    """This class utilizes the Augmented Lagrangian Uzawa method to solve
    the pressure and velocity of Darcy flow. For details, see the following:
    A new fast method to compute saddle-points in contrained optimizaiton and
    applications by Angot et. al., 2012. doi: 10.1016/j.aml.2011.08.015.
    """

    def __init__(self, mesh, boundary_markers, domain_markers):
        self.set_mesh(mesh)
        self.set_boundary_markers(boundary_markers)
        self.set_domain_markers(domain_markers)

        self.velocity_bc = []

    def set_pressure_ic(self, init_cond_pressure: Expression):
        """Sets up the initial condition of pressure."""
        self.init_cond_pressure = init_cond_pressure

    def mark_flow_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Keywords
        --------
        velocity: Sets the boundary flow rate.
        pressure: Sets the DirichletBC of pressure.
        """

        self.__boundary_dict = kwargs

    def set_form_and_pressure_bc(self, pressure_bc_val: list):
        """Sets up the FeNiCs form of Darcy flow."""

        V = self.velocity_func_space
        Q = self.pressure_func_space

        self.__u, self.__p = TrialFunction(V), TrialFunction(Q)
        self.__v, self.__q = TestFunction(V), TestFunction(Q)

        u, p = self.__u, self.__p
        v, q = self.__v, self.__q

        self.v0 = Function(V) # Test function used for residual calculations
        self.v0.vector()[:] = 1.0
        v0 = self.v0

        self.__u0, self.__u1 = Function(V), Function(V)
        self.__p0 = interpolate(self.init_cond_pressure, Q)
        self.__p1 = Function(Q)

        u0, u1, p0, p1 = self.__u0, self.__u1, self.__p0, self.__p1

        u0.rename('velocity', 'fluid velocity')
        p0.rename('pressure', 'fluid pressure')

        mu, k, rho, g = self._mu, self._k, self._rho, self._g

        self.r = Constant(0.0)
        r = self.r

        n = self.n
        dx, ds, dS = self.dx, self.ds, self.dS

        # AL2 prediction-correction scheme
        self.form_update_velocity_1 = mu/k*inner(v, u)*dx \
                                      - inner(v, rho*g)*dx
                                      #- inner(p0, div(v))*dx \

        #self.drho_dt = (self.rho - self.rho_old)/self.dt
        self.form_update_velocity_2 = mu/k*inner(v, u)*dx \
                                      + r*inner(div(v), div(u))*dx \
                                      + r*inner(div(v), div(u0))*dx
                                      #- r*inner(div(v), self.drho_dt)*dx

        # Pressure update
        #self.form_update_pressure = q*p*dx - q*p0*dx + r*q*(div(u1))*dx
        self.form_update_pressure = q*p*dx + r*q*div(u1)*dx

        self.residual_momentum_form = mu/k*inner(v0, u0)*dx \
                                      - inner(div(v0), p0)*dx \
                                      - inner(v0, rho*g)*dx
        self.residual_mass_form = q*div(u0)*dx

        for i, marker in enumerate(self.__boundary_dict['pressure']):
            self.form_update_velocity_1 += pressure_bc_val[i]*inner(n, v) \
                                           *ds(marker)

            #self.form_update_velocity_2 += pressure_bc_val[i]*inner(n, v) \
            #                               *ds(marker)

            self.residual_momentum_form += pressure_bc_val[i]*inner(n, v0) \
                                           *ds(marker)

    def set_angot_parameters(self, r_val: float):
        """r is 1/epsilon in the literature. r >> 1."""

        self.r.assign(Constant(r_val))

    def set_velocity_bc(self, velocity_bc_val: list):
        """
        Arguments
        ---------
        velocity_bc_val : list of Constants,
                          e.g., [Constant((1.0, -1.0)), Constant((0.0, -2.0))]
        """

        self.velocity_bc = []

        for i, marker in enumerate(self.__boundary_dict['velocity']):
            self.velocity_bc.append(DirichletBC(self.velocity_func_space,
                                                velocity_bc_val[i],
                                                self.boundary_markers, marker))

    def get_residual(self):
        """"""
        residual_momentum = assemble(self.residual_momentum_form)
        residual_mass = assemble(self.residual_mass_form)

        for bc in self.velocity_bc:
            bc.apply(self.__u0.vector())
            bc.apply(self.v0.vector())

        residual = abs(residual_momentum)
        residual += residual_mass.norm('l2')

        return residual

    def assemble_matrix(self):
        F_velocity_1 = self.form_update_velocity_1
        F_velocity_2 = self.form_update_velocity_2
        F_pressure = self.form_update_pressure

        a_v1, self.L_v1 = lhs(F_velocity_1), rhs(F_velocity_1)
        a_v2, self.L_v2 = lhs(F_velocity_2), rhs(F_velocity_2)
        a_p, self.L_p = lhs(F_pressure), rhs(F_pressure)

        self.A_v1 = assemble(a_v1)
        self.A_v2 = assemble(a_v2)
        self.A_p = assemble(a_p)
        self.b_v1, self.b_v2 = PETScVector(), PETScVector()
        self.b_p = PETScVector()

    def set_solver(self):
        # Users can override this method.
        # Or, TODO: make this method more user friendly.

        self.solver_v1 = PETScKrylovSolver('gmres', 'amg')
        self.solver_v2 = PETScLUSolver('mumps')
        self.solver_p = PETScKrylovSolver('gmres', 'amg')

        prm_v1 = self.solver_v1.parameters
        prm_v2 = self.solver_v2.parameters
        prm_p = self.solver_p.parameters

        TransportProblemBase.set_default_solver_parameters(prm_v1)
        TransportProblemBase.set_default_solver_parameters(prm_p)

    def solve_flow(self, target_residual: float, max_steps: int):
        steps = 0

        #while(self.get_residual() > target_residual and steps < max_steps):
        assemble(self.L_v1, tensor=self.b_v1)
        for bc in self.velocity_bc:
            bc.apply(self.A_v1, self.b_v1)

        self.solver_v1.solve(self.A_v1, self.__u0.vector(), self.b_v1)

        assemble(self.L_v2, tensor=self.b_v2)
        for bc in self.velocity_bc:
            bc.apply(self.A_v2, self.b_v2)

        self.solver_v2.solve(self.A_v2, self.__u1.vector(), self.b_v2)
        self.__u1.assign( (self.__u1 + self.__u0)/2 )

        assemble(self.L_p, tensor=self.b_p)
        self.solver_p.solve(self.A_p, self.__p1.vector(), self.b_p)

        self.__u0.assign(self.__u1)
        self.__p0.assign(self.__p1)

        steps+=1

        self.fluid_velocity.assign(self.__u0)
        self.fluid_pressure.assign(self.__p0)
