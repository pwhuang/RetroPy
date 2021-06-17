from . import *

class DarcyFlowAngot(TransportProblemBase, DarcyFlowBase):
    """This class utilizes the Augmented Lagrangian Uzawa method to solve
    the pressure and velocity of Darcy flow. For details, see the following:
    A new fast method to compute saddle-points in contrained optimizaiton and
    applications by Angot et. al., 2012. doi: 10.1016/j.aml.2011.08.015.
    """

    def __init__(self, mesh, boundary_markers, domain_markers):
        DarcyFlowBase.__init__(self, mesh, boundary_markers, domain_markers)

    def generate_form(self):
        """Sets up the FeNiCs form of Darcy flow."""

        V = self.velocity_func_space
        Q = self.pressure_func_space

        self.__u, self.__p = TrialFunction(V), TrialFunction(Q)
        self.__v, self.__q = TestFunction(V), TestFunction(Q)

        u, p = self.__u, self.__p
        v, q = self.__v, self.__q

        self.__u0, self.__u1 = Function(V), Function(V)
        self.__p0 = Function(Q)
        self.__p1 = Function(Q)

        u0, u1, p0, p1 = self.__u0, self.__u1, self.__p0, self.__p1

        mu, k, rho, g, phi = self._mu, self._k, self._rho, self._g, self._phi

        self.r = Constant(0.0)
        r = self.r

        n = self.n
        dx, ds, dS = self.dx, self.ds, self.dS

        # AL2 prediction-correction scheme
        self.form_update_velocity_1 = mu/k*inner(v, u)*dx \
                                      - inner(v, rho*g)*dx
                                      #- inner(p0, div(v))*dx

        #self.drho_dt = (self.rho - self.rho_old)/self.dt
        self.form_update_velocity_2 = mu/k*inner(v, u)*dx \
                                      + r*inner(div(v), div(rho*phi*u))*dx \
                                      + r*inner(div(v), div(rho*phi*u0))*dx
                                      #- r*inner(div(v), self.drho_dt)*dx

        # Pressure update
        #self.form_update_pressure = q*(p-p0)*dx+ r*q*(div(rho*phi*u1))*dx
        self.form_update_pressure = q*p*dx + r*q*div(rho*phi*u1)*dx

        for i, marker in enumerate(self.darcyflow_boundary_dict['pressure']):
            self.form_update_velocity_1 += self.pressure_bc[i]*inner(n, v) \
                                           *ds(marker)

            self.form_update_velocity_2 += self.pressure_bc[i]*inner(n, v) \
                                           *ds(marker)

    def add_momentum_source(self, sources: list):
        v = self.__v

        for source in sources:
            self.form_update_velocity_1 -= inner(v, source)*self.dx
            self.form_update_velocity_2 -= inner(v, source)*self.dx

    def set_angot_parameters(self, r_val: float):
        """r is 1/epsilon in the literature. r >> 1."""

        self.r.assign(Constant(r_val))

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

        self.solver_v1 = PETScLUSolver('mumps')
        self.solver_v2 = PETScLUSolver('mumps')
        self.solver_p = PETScLUSolver('mumps')
        #self.solver_v1 = PETScKrylovSolver('gmres', 'ilu')
        #self.solver_v2 = PETScKrylovSolver('gmres', 'ilu')
        #self.solver_p = PETScKrylovSolver('gmres', 'amg')

        prm_v1 = self.solver_v1.parameters
        prm_v2 = self.solver_v2.parameters
        prm_p = self.solver_p.parameters

        #TransportProblemBase.set_default_solver_parameters(prm_v1)
        #TransportProblemBase.set_default_solver_parameters(prm_v2)
        #TransportProblemBase.set_default_solver_parameters(prm_p)

    #def solve_flow(self, target_residual: float, max_steps: int):
    def solve_flow(self, *args):
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
        #self.__p1.vector()[:] = self.b_p.get_local()
        self.solver_p.solve(self.A_p, self.__p1.vector(), self.b_p)

        self.__u0.assign(self.__u1)
        self.__p0.assign(self.__p1)

        steps+=1

        self.fluid_velocity.assign(self.__u0)
        self.fluid_pressure.assign(self.__p0)
