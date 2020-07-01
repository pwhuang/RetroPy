import numpy as np
import reaktoro as rkt
from dolfin import *
from dolfin import sqrt, jump, avg
import dolfin
from ufl.classes import MaxValue
from ufl.algebra import Abs

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

parameters["ghost_mode"] = "shared_vertex"

import sys
sys.path.insert(0, '../../../Reaktoro-Transport')
from reaktoro_transport.solver import multicomponent_diffusion_problem
#from reaktoro_transport.solver import reactive_transport_problem_base
#from reaktoro_transport.solver.reactive_transport_problem_base import reactive_transport_problem_base
#import reaktoro_transport.tools as tools

class multicomponent_transport_problem(multicomponent_diffusion_problem):
    def __init__(self):
        super().__init__()
        self.xdmf_obj = dolfin.XDMFFile(MPI.comm_world, 'solution_output_primal.xdmf')

    def set_transport_species(self, num_transport_components, initial_expr):
        self.num_transport_components = num_transport_components

        self.function_space = dolfin.FunctionSpace(self.mesh, 'DG', 0)

        self.x_ = dolfin.interpolate(dolfin.Expression("x[0]", degree=1), self.function_space)
        self.y_ = dolfin.interpolate(dolfin.Expression("x[1]",degree=1), self.function_space)

        self.Delta_h = sqrt(jump(self.x_)**2 + jump(self.y_)**2)

        self.par_test = dolfin.Function(self.function_space)
        self.rho = dolfin.Function(self.function_space)
        self.K = dolfin.project(Constant(1.0/48), self.function_space)

        self.rho.rename('density', 'density of fluid')

        self.X_list = []      # Xn
        self.X_list_old = []  # Xn-1
        self.X_list_temp = [] # Temporary storage
        self.mu_list = []     # Chemical potential
        self.mu_list_temp = []

        for i in range(self.num_transport_components):
            self.X_list.append(dolfin.Function(self.function_space))
            self.X_list_old.append(dolfin.interpolate(initial_expr[i], self.function_space))
        #for i in range(self.num_components):
            self.mu_list.append(dolfin.Function(self.function_space))
            self.X_list_old[i].rename(self.component_list[i], 'mass fraction of species')

        self.num_dof = len(self.X_list[0].vector())
        self.max_dof = int(MPI.max(MPI.comm_world, self.num_dof))

        for i in range(self.num_transport_components):
            self.X_list_temp.append(np.zeros(self.num_dof))
            self.mu_list_temp.append(np.zeros(self.num_dof))

        self.rho_temp = np.zeros(self.num_dof)

        #print('max dof = ', MPI.max(MPI.comm_world, self.num_dof))
        #print('num_dof = ', self.num_dof, MPI.rank(MPI.comm_world))

        return self.X_list_old

    def set_transport_species_mixed(self, num_transport_components, initial_expr):
        self.num_transport_components = num_transport_components

        self.function_space = dolfin.FunctionSpace(self.mesh, 'DG', 0)

        self.x_ = dolfin.interpolate(dolfin.Expression("x[0]", degree=1), self.function_space)
        self.y_ = dolfin.interpolate(dolfin.Expression("x[1]",degree=1), self.function_space)

        self.Delta_h = sqrt(jump(self.x_)**2 + jump(self.y_)**2)

        self.par_test = dolfin.Function(self.function_space)
        self.rho = dolfin.Function(self.function_space)

        self.X_list = []      # Xn
        self.X_list_old = []  # Xn-1
        self.X_list_temp = [] # Temporary storage
        self.mu_list = []     # Chemical potential
        self.mu_list_temp = []

        # Implement the mixed version here?
        mix_element_list = []
        for i in range(self.num_transport_components):
            mix_element_list.append(FiniteElement('DG', self.mesh.cell_name(), 0))

        self.mixed_func_space = FunctionSpace(self.mesh, MixedElement(mix_element_list))

        self.num_dof = len(self.x_.vector())
        self.max_dof = int(MPI.max(MPI.comm_world, self.num_dof))

    def set_boundary_conditions(self):
        # The user should override this function to define boundary conditions!
        self.b_dict = {'inlet': [], 'noslip': [1, 2, 3, 4], }
        self.p_list = []
        self.bc_list = []
#         [dolfin.DirichletBC(self.function_space, dolfin.Constant(0.0), self.boundary_markers, 2),
#          dolfin.DirichletBC(self.function_space, dolfin.Constant(0.0), self.boundary_markers, 1)]

    def set_flow_equations(self, omega_num=1.0, r_num=0.0):

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

        #self.p0 = project(Expression('9.81*(50.0-x[1])-240.0', degree=1), Q)
        self.p0 = project(Expression('0.0', degree=1), Q)
        self.p0.rename('pressure', 'fluid pressure')
        self.p1 = Function(Q)

        g = as_vector([0.0, -9806.65])

        # Define coefficients
        f = Constant((0, 0))
        mu = Constant(8.9e-4)

        one = Constant(1.0)
        r = Constant(r_num)
        omega = Constant(omega_num)

        self.nn = FacetNormal(self.mesh)

        # Tentative velocity step
        F1 = mu/self.K*inner(v, u)*dx - inner(self.p0, div(v))*dx \
             - inner(v, self.rho*g)*dx \
             + r*inner(div(v), div(u))*dx # grad-div stabilization

        for i, p_dirichlet in enumerate(self.p_list):
             F1 += Constant(p_dirichlet)*inner(self.nn, v)*ds(boundary_dict['inlet'][i])

        a1 = lhs(F1)
        L1 = rhs(F1)

        self.solver_v = LinearVariationalSolver(LinearVariationalProblem(a1, L1, self.u1, bcs=self.bcu))

        # Pressure update
        a2 = q*p*dx
        L2 = q*self.p0*dx - omega*q*div(self.u1)*dx

        self.solver_p = LinearVariationalSolver(LinearVariationalProblem(a2, L2, self.p1, bcs=[]))

        res_list = []

        prm = self.solver_v.parameters

        prm['krylov_solver']['absolute_tolerance'] = 1e-13
        #prm['ksp_converged_reason'] = True
        prm['krylov_solver']['relative_tolerance'] = 1e-11
        prm['krylov_solver']['maximum_iterations'] = 20000
        prm['krylov_solver']['error_on_nonconvergence'] = True
        #prm['krylov_solver']['monitor_convergence'] = True
        prm['krylov_solver']['nonzero_initial_guess'] = False
        prm['linear_solver'] = 'gmres'
        prm['preconditioner'] = 'amg'

        prm = self.solver_p.parameters

        prm['krylov_solver']['absolute_tolerance'] = 1e-13
        #prm['ksp_converged_reason'] = True
        prm['krylov_solver']['relative_tolerance'] = 1e-11
        prm['krylov_solver']['maximum_iterations'] = 20000
        prm['krylov_solver']['error_on_nonconvergence'] = True
        #prm['krylov_solver']['monitor_convergence'] = True
        prm['krylov_solver']['nonzero_initial_guess'] = True
        prm['linear_solver'] = 'gmres'
        prm['preconditioner'] = 'amg'

    def solve_flow(self, max_steps=500, res_target=1e-12):
        residual = 1.0
        i = 0

        #for i in range(steps):
        while(np.abs(residual) > res_target):
            begin("Computing tentative velocity")
            self.solver_v.solve()
            end()

            # Pressure correction
            begin("Computing pressure correction, residual = " + str(residual))
            self.solver_p.solve()

            div_u = assemble(self.q0*div(self.u1)*dx).norm('l2')
            residual_form = (inner(self.v0, self.u1) - self.p1*div(self.v0))*dx
            for i, p_dirichlet in enumerate(self.p_list):
                 residual_form += Constant(p_dirichlet)*inner(self.nn, self.v0)*ds(self.boundary_dict['inlet'][i])

            residual = assemble(residual_form) + div_u
            end()

            self.u0.assign(self.u1)
            self.p0.assign(self.p1)

            i+=1
            #xdmf_obj.write(u0, i)
            #xdmf_obj.write(p0, i)

            if (i>=max_steps):
                if MPI.rank(MPI.comm_world)==0:
                    print('Reached maximum steps!')
                break


    def set_transport_equations(self):
        v = dolfin.TestFunction(self.function_space)
        u = dolfin.TrialFunction(self.function_space)

        self.BDM_space = FunctionSpace(self.mesh, 'BDM', 1)
        self.adv = Function(self.BDM_space)

        #self.u_n   = dolfin.Function(self.function_space)
        #self.mu    = dolfin.Function(self.function_space)
        R = 8.314 # Ideal gas constant
        n = FacetNormal(self.mesh)

        adv_np = ( dot ( self.adv, n ) + Abs ( dot ( self.adv, n ) ) ) / 2.0
        adv_nm = ( dot ( self.adv, n ) - Abs ( dot ( self.adv, n ) ) ) / 2.0

        dS = dolfin.Measure('dS', domain=self.mesh, subdomain_data=self.boundary_markers)
        ds = dolfin.Measure('ds', domain=self.mesh, subdomain_data=self.boundary_markers)
        dx = dolfin.Measure('dx', domain=self.mesh, subdomain_data=self.boundary_markers)

        # Placeholder for dt
        self.dt = dolfin.Constant(1.0)
        self.solver_list = []

        # The form of grad_psi/F, assuming the solvent has no charge, e.g. H2O(l).
        grad_psi = dolfin.Constant(0.0)*jump(self.mu_list[0], n)
        denom = dolfin.Constant(0.0)
        for i in range(self.num_transport_components):
            grad_psi += Constant(self.D_list[i]*self.z_list[i]/self.M_list[i])\
                         *avg(self.X_list_old[i])*jump(self.mu_list[i], n)

            denom += Constant(self.D_list[i]*self.z_list[i]**2/self.M_list[i])*avg(self.X_list_old[i])

        grad_psi = -grad_psi/denom

        a = v*u/self.dt*dx \
            + dot(jump(v), adv_np('+')*u('+') - adv_np('-')*u('-') )*dS(0)
        for i in range(self.num_transport_components):
            L = v*self.X_list_old[i]/self.dt*dx\
                - Constant(self.D_list[i]/(R*self.T))\
                 *avg(self.X_list_old[i])\
                 *dot(jump(self.mu_list[i], n) + Constant(self.z_list[i])*grad_psi, jump(v, n))/self.Delta_h*dS(0)
                #- 0.5*dot(jump(v), adv_np('+')*self.X_list_old[i]('+') + adv_nm('+')*self.X_list_old[i]('-') )*dS(0)

            linear_problem = dolfin.LinearVariationalProblem(a, L, self.X_list[i], bcs=self.bc_list)
            self.solver_list.append(dolfin.LinearVariationalSolver(linear_problem))

    def solve_chemical_equilibrium(self):
        # Another method that needs to be overridden
        self.mass_bal_violation = 0

        #if MPI.rank(MPI.comm_world)==0:
        for i in range(self.num_dof):

            # This makes assignment concurrent, need to find a better way to do this.
#             if i > self.num_dof-1:
#                 self.rho.vector()[0] = self.rho.vector()[0]

#                 for j in range(self.num_transport_components):
#                     self.mu_list[j].vector()[0] = self.mu_list[j].vector()[0]
#                     self.X_list[j].vector()[0] = self.X_list[j].vector()[0]

#                 continue

            solvent_mass = 1.0 # Initialize solvent mass fraction
            for j in range(self.num_transport_components):
                if self.X_list[j].vector()[i] < 0.0:
                    self.mass_bal_violation += 1
                    self.chem_state.setSpeciesMass(j, 1e-16, 'kg')
                else:
                    solvent_mass -= self.X_list[j].vector()[i]
                    self.chem_state.setSpeciesMass(j, self.X_list[j].vector()[i], 'kg')

            self.chem_state.setSpeciesMass(self.num_components-1, solvent_mass, 'kg')
            self.chem_equi_solver.solve(self.chem_state)

            chem_prop = self.chem_state.properties()

            self.rho_temp[i] = chem_prop.phaseDensities().val[0]*1e-6

            for j in range(self.num_transport_components):
                self.mu_list_temp[j][i] = chem_prop.chemicalPotentials().val[j]
                self.X_list_temp[j][i] = self.chem_state.speciesAmount(j, 'mol')*self.M_list[j]

#             self.rho.vector()[i] = chem_prop.phaseDensities().val[0]*1e-6

#             for j in range(self.num_transport_components):
#                 self.mu_list[j].vector()[i] = chem_prop.chemicalPotentials().val[j]
#                 self.X_list[j].vector()[i] = self.chem_state.speciesAmount(j, 'mol')*self.M_list[j]

            del chem_prop

        # Concurrent function assignment
        self.rho.vector()[:] = self.rho_temp

        for j in range(self.num_transport_components):
            self.mu_list[j].vector()[:] = self.mu_list_temp[j]
            self.X_list[j].vector()[:] = self.X_list_temp[j]


    def test_parallel(self):
        # Setting function values must be concurrent
#         for i in range(self.num_dof):
#             self.par_test.vector()[i] = MPI.rank(MPI.comm_world)

        rank_list = np.ones(self.num_dof)*MPI.rank(MPI.comm_world)
        self.par_test.vector().set_local(rank_list)

    #def solve_velocity(self):


    def solve(self, dt_num, timesteps):
        out_list = []
        rho_list = []
        p_list = []
        u_list = []

        #for i in range(self.num_transport_components):
        #    out_list.append([self.X_list_old[i].copy()])

        self.dt.assign(dolfin.Constant(dt_num))

        for i in range(timesteps):
            begin('timestep = ' + str(i))
            self.solve_chemical_equilibrium()

            self.solve_flow(max_steps=200, res_target=1e-9)

            self.adv.assign(self.u0)

            #rho_list.append(self.rho.copy())
            self.xdmf_obj.write(self.rho, i*dt_num)
            self.xdmf_obj.write(self.adv, i*dt_num)
            self.xdmf_obj.write(self.p0, i*dt_num)

            for j in range(self.num_transport_components):
                self.solver_list[j].solve()

                self.X_list_old[j].assign(self.X_list[j])
                self.xdmf_obj.write(self.X_list_old[j], i*dt_num)

                #out_list[j].append(self.X_list_old[j].copy())

            #MPI.barrier(MPI.comm_world)
            sum_violation = int(MPI.sum(MPI.comm_world, self.mass_bal_violation))
            begin('violation count = ' + str(sum_violation))
            if sum_violation == 0:
                dt_num = dt_num*1.0
                self.dt.assign(dolfin.Constant(dt_num))
            elif sum_violation > 0:
                dt_num = dt_num*0.8
                self.dt.assign(dolfin.Constant(dt_num))
            end()


        self.xdmf_obj.close()

        return out_list, u_list, p_list

    def output(self):
        xdmf_obj = dolfin.XDMFFile(MPI.comm_world, 'solution_output_primal.xdmf')
        xdmf_obj.write(self.par_test, 0)
        xdmf_obj.close()



mesh_2d = RectangleMesh.create(MPI.comm_world, [Point(0.0, 0.0), Point(31.0, 50.0)], [31, 60], CellType.Type.triangle, 'right/left')
boundary_markers = MeshFunction('size_t', mesh_2d, dim=1)

DG_space = FunctionSpace(mesh_2d, 'DG', 0)
rho = Function(DG_space)

boundary_markers.set_all(0)

#print('process started!')

class left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 31.0, DOLFIN_EPS)

class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, DOLFIN_EPS)

class top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 50.0, DOLFIN_EPS)

class boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

b_left = left()
b_right = right()
b_bottom = bottom()
b_top = top()
b_boundary = boundary()

# Boundary on the fluid domain is marked as 0
boundary_markers.set_all(0)

# Then Mark boundaries
b_right.mark(boundary_markers, 1)
b_top.mark(boundary_markers, 2)
b_left.mark(boundary_markers, 3)
b_bottom.mark(boundary_markers, 4)

problem = multicomponent_transport_problem()

pressure = 1.0 #atm
temperature = 273.15+25 #K
molar_mass  = [22.99e-3, 35.453e-3, 1e-3, 17e-3]
diffusivity = [1.33e-3, 2.03e-3, 9.31e-3, 5.28e-3]
charge      = [1.0, -1.0, 1.0, -1.0]

problem.set_chemical_system(['Na+', 'Cl-', 'H+', 'OH-', 'H2O(l)'],\
                            pressure, temperature,\
                            molar_mass, diffusivity, charge)

problem.set_mesh(mesh_2d, boundary_markers)

init_expr_list = [Expression('x[1]<=25 ? 0.040/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[3]/molar_mass[0]),\
                  Expression('x[1]>25 ? 0.03646/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[2]/molar_mass[1]),\
                  Expression('x[1]>25 ? 0.03646/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[1]/molar_mass[2]),\
                  Expression('x[1]<=25 ? 0.040/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[0]/molar_mass[3])]

dt_num = 2.5
timesteps = 50

problem.set_transport_species(4, init_expr_list)
problem.set_boundary_conditions()
problem.set_flow_equations(omega_num=80.0, r_num=50.0)
problem.set_transport_equations()

u_list_mult, u_list, p_list = problem.solve(dt_num, timesteps)

#problem.output()

#print(rho.vector().local_size())

print('mass_bal_violation = ', problem.mass_bal_violation, 'process = ', MPI.rank(MPI.comm_world))
