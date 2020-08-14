#from . import np, rkt
from . import *

class multicomponent_transport_problem(multicomponent_diffusion_problem):
    def __init__(self, output_file):
        super().__init__()
        self.xdmf_obj = dolfin.XDMFFile(MPI.comm_world, output_file)
        self.xdmf_obj.parameters['flush_output'] = True
        self.xdmf_obj.parameters['rewrite_function_mesh'] = False
        # self.xdmf_obj = dolfin.HDF5File(MPI.comm_world, 'solution_output_primal.h5', 'w')
        # Placeholder for dt
        self.dt = dolfin.Constant(1.0)

    def set_transport_species(self, num_transport_components, initial_expr):
        self.num_transport_components = num_transport_components

        self.function_space = dolfin.FunctionSpace(self.mesh, 'DG', 0)

        self.x_ = dolfin.interpolate(dolfin.Expression("x[0]", degree=1), self.function_space)
        self.y_ = dolfin.interpolate(dolfin.Expression("x[1]",degree=1), self.function_space)

        self.Delta_h = sqrt(jump(self.x_)**2 + jump(self.y_)**2)

        #self.par_test = dolfin.Function(self.function_space)
        self.rho = dolfin.Function(self.function_space)
        self.rho_old = dolfin.Function(self.function_space)
        self.charge_func = dolfin.Function(self.function_space)
        self.pH_func = dolfin.Function(self.function_space)

        self.rho.rename('density', 'density of fluid')
        self.charge_func.rename('charge', 'charge of fluid')
        self.pH_func.rename('pH', 'pH of fluid')

        self.X_list = []      # Xn
        self.X_list_old = []  # Xn-1
        self.X_list_temp = [] # Temporary storage
        self.mu_list = []     # Chemical potential
        self.mu_list_temp = []

        for i in range(self.num_transport_components):
            self.X_list.append(dolfin.interpolate(initial_expr[i], self.function_space))
            self.X_list_old.append(dolfin.interpolate(initial_expr[i], self.function_space))
        #for i in range(self.num_components):
            self.mu_list.append(dolfin.Function(self.function_space))
            #self.X_list[i].rename(self.component_list[i], 'mass fraction of species')
            self.X_list_old[i].rename(self.component_list[i], 'mass fraction of species')

        self.num_dof = len(self.X_list[0].vector())
        self.max_dof = int(MPI.max(MPI.comm_world, self.num_dof))

        for i in range(self.num_transport_components):
            self.X_list_temp.append(np.zeros(self.num_dof))
            self.mu_list_temp.append(np.zeros(self.num_dof))

        self.rho_temp = np.zeros(self.num_dof)
        self.charge_func_temp = np.zeros(self.num_dof)
        self.pH_func_temp = np.zeros(self.num_dof)

        #print('max dof = ', MPI.max(MPI.comm_world, self.num_dof))
        #print('num_dof = ', self.num_dof, MPI.rank(MPI.comm_world))

        #return self.X_list_old

    # def set_transport_species_mixed(self, num_transport_components, initial_expr):
    #     self.num_transport_components = num_transport_components
    #
    #     self.function_space = dolfin.FunctionSpace(self.mesh, 'DG', 0)
    #
    #     self.x_ = dolfin.interpolate(dolfin.Expression("x[0]", degree=1), self.function_space)
    #     self.y_ = dolfin.interpolate(dolfin.Expression("x[1]",degree=1), self.function_space)
    #
    #     self.Delta_h = sqrt(jump(self.x_)**2 + jump(self.y_)**2)
    #
    #     self.par_test = dolfin.Function(self.function_space)
    #     self.rho = dolfin.Function(self.function_space)
    #     self.rho_old = dolfin.Function(self.function_space)
    #     self.charge_func = dolfin.Function(self.function_space)
    #
    #     self.X_list = []      # Xn
    #     self.X_list_old = []  # Xn-1
    #     self.X_list_temp = [] # Temporary storage
    #     self.mu_list = []     # Chemical potential
    #     self.mu_list_temp = []
    #
    #     # Implement the mixed version here?
    #     mix_element_list = []
    #     for i in range(self.num_transport_components):
    #         mix_element_list.append(FiniteElement('DG', self.mesh.cell_name(), 0))
    #
    #     self.mixed_func_space = FunctionSpace(self.mesh, MixedElement(mix_element_list))
    #
    #     self.num_dof = len(self.x_.vector())
    #     self.max_dof = int(MPI.max(MPI.comm_world, self.num_dof))

    def set_boundary_conditions(self):
        # The user should override this function to define boundary conditions!
        self.b_dict = {'inlet': [], 'noslip': [1, 2, 3, 4], }
        self.p_list = []
        self.bc_list = []
#         [dolfin.DirichletBC(self.function_space, dolfin.Constant(0.0), self.boundary_markers, 2),
#          dolfin.DirichletBC(self.function_space, dolfin.Constant(0.0), self.boundary_markers, 1)]

    def set_flow_equations(self):
        RT = dolfin.FiniteElement('BDM', self.mesh.cell_name(), 1)
        #RT = VectorElement('CR', mesh.cell_name(), 1)
        DG = dolfin.FiniteElement('DG', self.mesh.cell_name(), 0)

        DG_space = dolfin.FunctionSpace(self.mesh, 'DG', 0)

        W = dolfin.FunctionSpace(self.mesh, MixedElement([RT, DG]))
        self.U0 = dolfin.Function(W)

        phi = Constant(1.0)
        K = Constant(1.0/48)  # aperture = 0.5 mm
        c = Constant(4e-10)   # Pa^-1
        mu = Constant(8.9e-4)
        g = dolfin.as_vector([0.0, -9806.65])

        (u, p) = dolfin.TrialFunctions(W)
        (v, q) = dolfin.TestFunctions(W)

        n = dolfin.FacetNormal(self.mesh)

        zero = Constant((0.0, 0.0))

        ds = dolfin.Measure('ds', domain=self.mesh, subdomain_data=self.boundary_markers)
        dS = dolfin.Measure('dS', domain=self.mesh, subdomain_data=self.boundary_markers)

        u_list = []
        bc_list = [dolfin.DirichletBC(W.sub(0), zero, self.boundary_markers, 1),
                   dolfin.DirichletBC(W.sub(0), zero, self.boundary_markers, 3),
                   dolfin.DirichletBC(W.sub(0), zero, self.boundary_markers, 4),
                   dolfin.DirichletBC(W.sub(0), zero, self.boundary_markers, 2)]

        #self.drho_dt = (self.rho - self.rho_old)/self.dt

        F = mu/K*inner(v, u)*dx - inner(div(v), p)*dx - inner(v, self.rho*g)*dx \
            + q*div(self.rho*u)*dx # + q*self.drho_dt*dx
            #+ p_b*inner(n, v)*ds(4)
            #+ q*rho*phi*c*(p-U0[1])/dt_num*dx
            #- p_b*inner(n, v)*ds(3)\
            #- p_b*inner(n, v)*ds(2)

        a, L = dolfin.lhs(F), dolfin.rhs(F)

        problem = dolfin.LinearVariationalProblem(a, L, self.U0, bcs=bc_list)
        self.flow_solver = dolfin.LinearVariationalSolver(problem)

        prm = self.flow_solver.parameters

        prm['krylov_solver']['absolute_tolerance'] = 1e-15
        prm['krylov_solver']['relative_tolerance'] = 1e-13
        prm['krylov_solver']['maximum_iterations'] = 3000
        prm['krylov_solver']['monitor_convergence'] = False
        #if iterative_solver:
        prm['linear_solver'] = 'mumps'
        prm['preconditioner'] = 'none'

    def solve_flow(self):
        self.flow_solver.solve()
        #self.u0, self.p0 = self.U0.split(True)

    def set_transport_equations(self):
        v = dolfin.TestFunction(self.function_space)
        u = dolfin.TrialFunction(self.function_space)

        self.BDM_space = dolfin.FunctionSpace(self.mesh, 'BDM', 1)
        self.adv = dolfin.Function(self.BDM_space)
        self.adv.rename('velocity', 'velocity field')

        #self.u_n   = dolfin.Function(self.function_space)
        #self.mu    = dolfin.Function(self.function_space)
        R = 8.314 # Ideal gas constant
        n = dolfin.FacetNormal(self.mesh)

        adv_np = ( dot ( self.adv, n ) + Abs ( dot ( self.adv, n ) ) ) / 2.0
        adv_nm = ( dot ( self.adv, n ) - Abs ( dot ( self.adv, n ) ) ) / 2.0

        dS = dolfin.Measure('dS', domain=self.mesh, subdomain_data=self.boundary_markers)
        ds = dolfin.Measure('ds', domain=self.mesh, subdomain_data=self.boundary_markers)
        dx = dolfin.Measure('dx', domain=self.mesh, subdomain_data=self.boundary_markers)

        self.solver_list = []

        # The form of grad_psi/F, assuming the solvent has no charge, e.g. H2O(l).
        grad_psi = dolfin.Constant(0.0)*jump(self.mu_list[0])
        denom = dolfin.Constant(0.0)
        for i in range(self.num_transport_components):
            grad_psi += Constant(self.D_list[i]*self.z_list[i]/self.M_list[i])\
                         *avg(self.X_list_old[i])*jump(self.mu_list[i])

            denom += Constant(self.D_list[i]*self.z_list[i]**2/self.M_list[i])*avg(self.X_list_old[i])

        grad_psi = -grad_psi/denom

        a = v*u/self.dt*dx \
            + dot(jump(v), adv_np('+')*u('+') - adv_np('-')*u('-') )*dS(0)
        for i in range(self.num_transport_components):
            L = v*self.X_list_old[i]/self.dt*dx\
                - Constant(self.D_list[i]/(R*self.T))\
                 *avg(self.X_list_old[i])\
                 *dot(jump(self.mu_list[i]) + Constant(self.z_list[i])*grad_psi, jump(v))/self.Delta_h*dS(0)
                 #- 0.5*dot(jump(v), adv_np('+')*self.X_list_old[i]('+') - adv_np('-')*self.X_list_old[i]('-') )*dS(0)
                 #*dot(jump(self.mu_list[i]) + Constant(self.z_list[i])*grad_psi, jump(v))/self.Delta_h*dS(0)

            linear_problem = dolfin.LinearVariationalProblem(a, L, self.X_list[i], bcs=self.bc_list)
            self.solver_list.append(dolfin.LinearVariationalSolver(linear_problem))

    def solve_chemical_equilibrium(self):
        # Takes in X_list
        # Another method that needs to be overridden
        self.mass_bal_violation = 0

        #if MPI.rank(MPI.comm_world)==0:
        for i in range(self.num_dof):

            solvent_mass = 1.0 # Initialize solvent mass fraction
            for j in range(self.num_transport_components):
                if self.X_list[j].vector()[i] < 0.0:
                    self.mass_bal_violation += 1
                    self.chem_state.setSpeciesMass(j, 1e-12, 'kg')
                else:
                    solvent_mass -= self.X_list[j].vector()[i]
                    self.chem_state.setSpeciesMass(j, self.X_list[j].vector()[i], 'kg')

            if solvent_mass < 0.0:
                self.mass_bal_violation = 12345
                break

            self.chem_state.setSpeciesMass(self.num_components-1, solvent_mass, 'kg')
            self.chem_state.setPressure(self.p0.vector()[i], 'Pa')

            self.chem_equi_solver.solve(self.chem_state)

            #chem_prop = self.chem_state.properties()
            self.chem_quant = self.chem_quant.update(self.chem_state)
            #self.chem_quant = rkt.ChemicalQuantity(self.chem_state)

            self.rho_temp[i] = self.chem_quant('phaseMass(Aqueous)')/self.chem_quant('fluidVolume(units==m3)')*1e-6
            self.pH_func_temp[i] = self.chem_quant('pH')
            #self.rho_temp[i] = chem_prop.phaseDensities().val[0]*1e-6

            self.charge_func_temp[i] = 0.0

            for j in range(self.num_transport_components):
                self.mu_list_temp[j][i] = self.chem_quant('chemicalPotential(' + self.component_list[j] + ')')
                #self.mu_list_temp[j][i] = chem_prop.chemicalPotentials().val[j]
                #self.X_list_temp[j][i] = self.chem_state.speciesAmount(j, 'mol')*self.M_list[j]
                self.X_list_temp[j][i] = self.chem_quant('speciesMass(' + self.component_list[j] + ' units=kg)')
                self.charge_func_temp[i] += self.chem_state.speciesAmount(j, 'mol')*self.z_list[j]

            # self.rho.vector()[i] = chem_prop.phaseDensities().val[0]*1e-6

#             for j in range(self.num_transport_components):
#                 self.mu_list[j].vector()[i] = chem_prop.chemicalPotentials().val[j]
#                 self.X_list[j].vector()[i] = self.chem_state.speciesAmount(j, 'mol')*self.M_list[j]

        #
        # Concurrent function assignment
        self.rho.vector()[:] = self.rho_temp
        self.pH_func.vector()[:] = self.pH_func_temp
        self.charge_func.vector()[:] = self.charge_func_temp

        for j in range(self.num_transport_components):
            self.mu_list[j].vector()[:] = self.mu_list_temp[j]
            self.X_list_old[j].vector()[:] = self.X_list_temp[j]


    def test_parallel(self):
        # Setting function values must be concurrent
#         for i in range(self.num_dof):
#             self.par_test.vector()[i] = MPI.rank(MPI.comm_world)

        rank_list = np.ones(self.num_dof)*MPI.rank(MPI.comm_world)
        self.par_test.vector().set_local(rank_list)

    #def solve_velocity(self):

    def solve(self, dt_num, dt_end):
        out_list = []
        for i in range(self.num_transport_components):
            out_list.append([self.X_list_old[i].copy()])

        self.dt.assign(Constant(dt_num))

        i = 0
        current_time = -dt_num # logic flowing everywhere! bad!

        while(current_time < dt_end):
            begin('timestep = ' + str(i) + '  dt_num = ' + str(np.round(dt_num, 5))\
                   + '  current_time = ' + str(np.round(current_time + dt_num, 5)))

            # Solve transport and flow, assuming the input is already in chemical equilibrium

            # solve_flow takes rho
            self.solve_flow()

            self.rho_old.assign(self.rho)
            self.adv.assign(self.u0)
            mass_bal_violation = 0

            # Performing mass balance check
            for j in range(self.num_transport_components):
                self.solver_list[j].solve()

                mass_bal_violation += int(MPI.sum(MPI.comm_world, np.sum(self.X_list[j].vector()[:] < 0.0)))

            begin('violation count = ' + str(mass_bal_violation))
            end()

            if mass_bal_violation > 0:
                dt_num = dt_num*0.33
                self.dt.assign(Constant(dt_num))
                continue

            current_time += dt_num

            i+=1
            dt_num = dt_num*1.1
            if dt_num > 2.0:
                dt_num = 2.0

            self.dt.assign(Constant(dt_num))

            self.solve_chemical_equilibrium()
            self.xdmf_obj.write(self.charge_func, current_time)
            self.xdmf_obj.write(self.adv, current_time)

            for j in range(self.num_transport_components):
                #self.X_list_old[j].assign(self.X_list[j])
                self.xdmf_obj.write(self.X_list_old[j], current_time)
                #out_list[j].append(self.X_list_old[j].copy())

            end()

    def output(self):
        xdmf_obj = dolfin.XDMFFile(MPI.comm_world, 'solution_output_primal.xdmf')
        xdmf_obj.write(self.par_test, 0)
        xdmf_obj.close()
