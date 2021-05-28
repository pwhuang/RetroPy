from . import *

class DarcyFlowUzawa(TransportProblemBase):
    """This class utilizes the Augmented Lagrangian Uzawa method to solve
    the pressure and velocity of Darcy's flow.
    """

    def set_pressure_ic(self, init_cond_pressure: Expression):
        """Sets up the initial condition of pressure."""
        self.init_cond_pressure = init_cond_pressure

    def mark_flow_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Keywords
        --------
        inlet : Sets the boundary flow rate.
        noflow: Sets the boundary flow rate to zero.
        """

        self.__boundary_dict = kwargs

    def set_boundary_conditions(self):
        """"""
        self.b_dict = {'inlet': [], 'noflow': [1, 2, 3, 4], }

    def set_flow_equations(self, r_num=10.0):

        V = FunctionSpace(self.mesh, "BDM", 1)
        Q = FunctionSpace(self.mesh, "DG", 0)

        # Define trial and test functions
        u = TrialFunction(V)
        p = TrialFunction(Q)
        v = TestFunction(V)
        q = TestFunction(Q)
        self.q0 = TestFunction(Q)

        ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundary_markers)
        dS = Measure('dS', domain=self.mesh, subdomain_data=self.boundary_markers)

        self.bcu = []

        if self.mesh.geometric_dimension()==2:
            noslip = (0.0, 0.0)
        elif self.mesh.geometric_dimension()==3:
            noslip = (0.0, 0.0, 0.0)

        for idx in self.b_dict['noslip']:
            self.bcu.append(DirichletBC(V, noslip, self.boundary_markers, idx))

        # Create functions
        self.u0 = Function(V)
        self.u1 = Function(V)
        self.v0 = Function(V) # Test function used for residual calculations

        for bc in self.bcu:
            bc.apply(self.v0.vector())

        self.p_ref = 101550
        self.p0 = project(Expression('9.81*(25.0-x[1]) + pref', degree=1, pref = self.p_ref), Q)
        self.p0.rename('pressure', 'fluid pressure')
        self.p1 = Function(Q)

        self.K = dolfin.project(Constant(0.5**2/12.0), self.function_space)
        self.g = as_vector([0.0, -9806.65])

        # Define coefficients
        f = Constant((0, 0))
        self.mu = Constant(8.9e-4)

        one = Constant(1.0)
        r = Constant(r_num)
        #omega = Constant(omega_num)

        self.nn = FacetNormal(self.mesh)

        self.drho_dt = (self.rho - self.rho_old)/self.dt

        # AL2 prediction-correction scheme
        F0 = self.mu/self.K*inner(v, u)*dx - inner(self.p0, div(v))*dx \
             - inner(v, self.rho*self.g)*dx \

        for i, p_dirichlet in enumerate(self.p_list):
             F0 += Constant(p_dirichlet)*inner(self.nn, v)*ds(boundary_dict['inlet'][i])

        a0 = lhs(F0)
        L0 = rhs(F0)

        self.solver_v0 = LinearVariationalSolver(LinearVariationalProblem(a0, L0, self.u0, bcs=self.bcu))

        # Tentative velocity step
        F1 = self.mu/self.K*inner(v, u)*dx + r*inner(div(v), div(u))*dx \
             + r*inner(div(v), div(self.u0))*dx #- r*inner(div(v), self.drho_dt)*dx

        for i, p_dirichlet in enumerate(self.p_list):
             F1 += Constant(p_dirichlet)*inner(self.nn, v)*ds(boundary_dict['inlet'][i])

        a1 = lhs(F1)
        L1 = rhs(F1)

        self.solver_v1 = LinearVariationalSolver(LinearVariationalProblem(a1, L1, self.u1, bcs=self.bcu))

        # Pressure update
        a2 = q*p*dx
        L2 = q*self.p0*dx - r*q*(div(self.u1))*dx

        self.solver_p = LinearVariationalSolver(LinearVariationalProblem(a2, L2, self.p1, bcs=[]))

        res_list = []

        prm = self.solver_v0.parameters

        prm['krylov_solver']['absolute_tolerance'] = 1e-15
        #prm['ksp_converged_reason'] = True
        prm['krylov_solver']['relative_tolerance'] = 1e-13
        prm['krylov_solver']['maximum_iterations'] = 2000
        prm['krylov_solver']['error_on_nonconvergence'] = True
        #prm['krylov_solver']['monitor_convergence'] = True
        prm['krylov_solver']['nonzero_initial_guess'] = False
        prm['linear_solver'] = 'gmres'
        prm['preconditioner'] = 'amg'

        prm = self.solver_v1.parameters

        prm['krylov_solver']['absolute_tolerance'] = 1e-14
        #prm['ksp_converged_reason'] = True
        prm['krylov_solver']['relative_tolerance'] = 1e-12
        prm['krylov_solver']['maximum_iterations'] = 50000
        prm['krylov_solver']['error_on_nonconvergence'] = True
        #prm['krylov_solver']['monitor_convergence'] = True
        prm['krylov_solver']['nonzero_initial_guess'] = False
        prm['linear_solver'] = 'minres'
        prm['preconditioner'] = 'jacobi'

        prm = self.solver_p.parameters

        prm['krylov_solver']['absolute_tolerance'] = 1e-15
        #prm['ksp_converged_reason'] = True
        prm['krylov_solver']['relative_tolerance'] = 1e-13
        prm['krylov_solver']['maximum_iterations'] = 2000
        prm['krylov_solver']['error_on_nonconvergence'] = True
        #prm['krylov_solver']['monitor_convergence'] = True
        prm['krylov_solver']['nonzero_initial_guess'] = False
        prm['linear_solver'] = 'gmres'
        prm['preconditioner'] = 'amg'

    def solve_flow(self, max_steps=50, res_target=1e-10):
        residual = 1.0
        i = 0

        #for i in range(steps):
        while(np.abs(residual) > res_target):
            #begin("Computing tentative velocity")
            self.solver_v0.solve()
            self.solver_v1.solve()
            self.u1.assign(self.u1 + self.u0)
            #end()

            # Pressure correction
            begin("Computing pressure correction, residual = " + str(residual))
            self.solver_p.solve()

            div_u = assemble(self.q0*div(self.rho*self.u1)*dx ).norm('l2')
            residual_form = (self.mu/self.K*inner(self.v0, self.u1) - self.p1*div(self.v0) \
                             - inner(self.v0, self.rho*self.g) )*dx
            for i, p_dirichlet in enumerate(self.p_list):
                 residual_form += Constant(p_dirichlet)*inner(self.nn, self.v0)*ds(self.boundary_dict['inlet'][i])

            residual = assemble(residual_form) + div_u
            end()

            self.u0.assign(self.u1)
            self.p0.assign(self.p1)
