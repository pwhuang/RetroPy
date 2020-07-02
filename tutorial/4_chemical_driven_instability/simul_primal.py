import numpy as np
import reaktoro as rkt
from dolfin import *
import dolfin

parameters["ghost_mode"] = "shared_vertex"

import sys
sys.path.insert(0, '../../../Reaktoro-Transport')
from reaktoro_transport.solver import multicomponent_transport_problem

class multicomponent_transport_problem_uzawa(multicomponent_transport_problem):
    def set_boundary_conditions(self):
        # The user should override this function to define boundary conditions!
        self.b_dict = {'inlet': [], 'noslip': [1, 2, 3, 4], }
        self.p_list = []
        self.bc_list = []

    def set_flow_equations(self, r_num=0.0):

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
        #omega = Constant(omega_num)

        self.nn = FacetNormal(self.mesh)

        # AL2 prediction-correction scheme
        F0 = mu/self.K*inner(v, u)*dx - inner(self.p0, div(v))*dx \
             - inner(v, self.rho*g)*dx \

        for i, p_dirichlet in enumerate(self.p_list):
             F0 += Constant(p_dirichlet)*inner(self.nn, v)*ds(boundary_dict['inlet'][i])

        a0 = lhs(F0)
        L0 = rhs(F0)

        self.solver_v0 = LinearVariationalSolver(LinearVariationalProblem(a0, L0, self.u0, bcs=self.bcu))

        # Tentative velocity step
        F1 = mu/self.K*inner(v, u)*dx + r*inner(div(v), div(u))*dx \
             + r*inner(div(v), div(self.u0))*dx

        for i, p_dirichlet in enumerate(self.p_list):
             F1 += Constant(p_dirichlet)*inner(self.nn, v)*ds(boundary_dict['inlet'][i])

        a1 = lhs(F1)
        L1 = rhs(F1)

        self.solver_v1 = LinearVariationalSolver(LinearVariationalProblem(a1, L1, self.u1, bcs=self.bcu))

        # Pressure update
        a2 = q*p*dx
        L2 = q*self.p0*dx - r*q*div(self.u1)*dx

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
        prm['krylov_solver']['nonzero_initial_guess'] = True
        prm['linear_solver'] = 'gmres'
        prm['preconditioner'] = 'amg'

    def solve_flow(self, max_steps=50, res_target=1e-11):
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

            div_u = assemble(self.q0*div(self.u1)*dx).norm('l2')
            residual_form = (inner(self.v0, self.u1) - self.p1*div(self.v0))*dx
            for i, p_dirichlet in enumerate(self.p_list):
                 residual_form += Constant(p_dirichlet)*inner(self.nn, self.v0)*ds(self.boundary_dict['inlet'][i])

            residual = assemble(residual_form) + div_u
            end()

            self.u0.assign(self.u1)
            self.p0.assign(self.p1)

            i+=1

            if (i>=max_steps):
                if MPI.rank(MPI.comm_world)==0:
                    print('Reached maximum steps!')
                break

    def solve(self, dt_num, dt_end):

        self.dt.assign(dt_num)

        i = 0
        current_time = -dt_num # logic flowing everywhere! bad!

        while(current_time < dt_end):
            begin('timestep = ' + str(i) + '  dt_num = ' + str(np.round(dt_num, 5))\
                   + '  current_time = ' + str(np.round(current_time + dt_num, 5)))

            self.solve_chemical_equilibrium()

            sum_violation = int(MPI.sum(MPI.comm_world, self.mass_bal_violation))
            begin('violation count = ' + str(sum_violation))
            end()
            if sum_violation == 0:

                current_time += dt_num

                if i==0:
                    self.rho_old.assign(self.rho)

                self.dt.assign(dt_num)

                self.xdmf_obj.write(self.adv, current_time)
                self.xdmf_obj.write(self.rho, current_time)
                self.xdmf_obj.write(self.p0, current_time)

                # When there are no violations, overwrite X_list_old
                for j in range(self.num_transport_components):
                    self.X_list_old[j].assign(self.X_list[j])
                    self.xdmf_obj.write(self.X_list[j], current_time)

                i+=1
                dt_num = dt_num*1.1

            # When violations exist, lower dt_num then solve again!
            elif sum_violation > 0:
                dt_num = dt_num*0.33
                self.dt.assign(dt_num)

            end()

            # solve_flow takes rho
            self.solve_flow()

            self.rho_old.assign(self.rho)
            self.adv.assign(self.u0)

            for j in range(self.num_transport_components):
                self.solver_list[j].solve()


mesh_2d = RectangleMesh.create(MPI.comm_world, [Point(0.0, 0.0), Point(30.0, 30.0)], [30, 30], CellType.Type.triangle, 'right/left')
#mesh_2d = RectangleMesh.create(MPI.comm_world, [Point(0.0, 0.0), Point(31.0, 50.0)], [31, 60], CellType.Type.quadrilateral)
cell_markers = MeshFunction('bool', mesh_2d, dim=2)

class middle(SubDomain):
    def inside(self, x, on_boundary):
        return x[1]<17.5 and x[1]>12.5

# Refine middle part of the mesh
c_middle = middle()

cell_markers.set_all(0)
c_middle.mark(cell_markers, 1)

mesh_2d = refine(mesh_2d, cell_markers)

boundary_markers = MeshFunction('size_t', mesh_2d, dim=1)

boundary_markers.set_all(0)

#print('process started!')

class left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 30.0, DOLFIN_EPS)

class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, DOLFIN_EPS)

class top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 30.0, DOLFIN_EPS)

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

#set_log_active(20)

problem = multicomponent_transport_problem_uzawa('solution_output_primal.xdmf')

pressure = 1.0 #atm
temperature = 273.15+25 #K
molar_mass  = [22.99e-3, 35.453e-3, 1e-3, 17e-3]
diffusivity = [1.33e-3, 2.03e-3, 9.31e-3, 5.28e-3]
charge      = [1.0, -1.0, 1.0, -1.0]

problem.set_chemical_system(['Na+', 'Cl-', 'H+', 'OH-', 'H2O(l)'],\
                            pressure, temperature,\
                            molar_mass, diffusivity, charge)

problem.set_mesh(mesh_2d, boundary_markers)

init_expr_list = [Expression('x[1]<=15 ? 0.040/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[3]/molar_mass[0]),\
                  Expression('x[1]>15 ? 0.03646/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[2]/molar_mass[1]),\
                  Expression('x[1]>15 ? 0.03646/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[1]/molar_mass[2]),\
                  Expression('x[1]<=15 ? 0.040/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[0]/molar_mass[3])]

dt_num = 1.0
timesteps = 20.0

problem.set_transport_species(4, init_expr_list)
problem.set_boundary_conditions()
problem.set_flow_equations(r_num=1000.0)
problem.set_transport_equations()

problem.solve(dt_num, timesteps)

#problem.output()

#print(rho.vector().local_size())

print('mass_bal_violation = ', problem.mass_bal_violation, 'process = ', MPI.rank(MPI.comm_world))
