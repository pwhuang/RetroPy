import numpy as np
import reaktoro as rkt
import dolfin
from dolfin import *

parameters["ghost_mode"] = "shared_vertex"

import sys
sys.path.insert(0, '../../../Reaktoro-Transport')
from reaktoro_transport.solver import multicomponent_transport_problem
from reaktoro_transport.tools import get_mass_fraction

class multicomponent_transport_problem_uzawa(multicomponent_transport_problem):
    def set_boundary_conditions(self):
        # The user should override this function to define boundary conditions!
        self.b_dict = {'inlet': [], 'noslip': [1, 2, 3, 4], }
        self.p_list = []
        self.bc_list = []

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

            i+=1

            if (i>=max_steps):
                if MPI.rank(MPI.comm_world)==0:
                    print('Reached maximum steps!')
                break

    def solve(self, dt_num, dt_end):
        out_list = []
        #for i in range(self.num_transport_components):
        #    out_list.append([self.X_list_old[i].copy()])

        self.dt.assign(Constant(dt_num))

        i = 0
        current_time = 0.0 # logic flowing everywhere! bad!

        while(current_time < dt_end):
            begin('timestep = ' + str(i) + '  dt_num = ' + str(np.round(dt_num, 5))\
                   + '  current_time = ' + str(np.round(current_time + dt_num, 5)))

            # Solve transport and flow, assuming the input is already in chemical equilibrium
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
            dt_num = dt_num*1.2
            if dt_num > 2.0:
                dt_num = 2.0

            self.dt.assign(Constant(dt_num))

            self.solve_chemical_equilibrium()

            # solve_flow takes rho
            self.solve_flow()
            #self.rho_old.assign(self.rho)
            self.adv.assign(self.u0)

            self.xdmf_obj.write(self.charge_func, current_time)
            self.xdmf_obj.write(self.adv, current_time)
            self.xdmf_obj.write(self.p0, current_time)

            for j in range(self.num_transport_components):
                #self.X_list_old[j].assign(self.X_list[j])
                self.xdmf_obj.write(self.X_list_old[j], current_time)
                #out_list[j].append(self.X_list_old[j].copy())

            end()


mesh_2d = RectangleMesh.create(MPI.comm_world, [Point(0.0, 0.0), Point(31.0, 50.0)], [30, 40], CellType.Type.triangle, 'right/left')
#mesh_2d = RectangleMesh.create(MPI.comm_world, [Point(0.0, 0.0), Point(31.0, 50.0)], [31, 60], CellType.Type.quadrilateral)
cell_markers = MeshFunction('bool', mesh_2d, dim=2)

class middle(SubDomain):
    def inside(self, x, on_boundary):
        return x[1]<35.0 and x[1]>22.0

# Refine middle part of the mesh
c_middle = middle()

cell_markers.set_all(0)
c_middle.mark(cell_markers, 1)

mesh_2d = refine(mesh_2d, cell_markers)

boundary_markers = MeshFunction('size_t', mesh_2d, mesh_2d.topology().dim() - 1)

boundary_markers.set_all(0)

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

b_left = left()
b_right = right()
b_bottom = bottom()
b_top = top()

# Then Mark boundaries
b_right.mark(boundary_markers, 1)
b_top.mark(boundary_markers, 2)
b_left.mark(boundary_markers, 3)
b_bottom.mark(boundary_markers, 4)

#set_log_active(20)

problem = multicomponent_transport_problem_uzawa('solution_output_primal.xdmf')

pressure = 1.0 #atm
temperature = 273.15+25 #K
component_list = ['Na+', 'Cl-', 'H+', 'OH-', 'H2O(l)']
molar_mass  = [22.99e-3, 35.453e-3, 1.0e-3, 17.0e-3, 18.0e-3]
diffusivity = [1.33e-3, 2.03e-3, 9.31e-3, 5.28e-3]
charge      = [1.0, -1.0, 1.0, -1.0]

HCl_species_amounts = [1e-13, 1.0, 1.0, 1e-13, 54.17]
mass_frac_HCl = get_mass_fraction(component_list, pressure, temperature, molar_mass, HCl_species_amounts)

NaOH_species_amounts = [1.0, 1e-13, 1e-13, 1.0, 55.36]
mass_frac_NaOH = get_mass_fraction(component_list, pressure, temperature, molar_mass, NaOH_species_amounts)

problem.set_chemical_system(component_list,\
                            pressure, temperature,\
                            molar_mass, diffusivity, charge)

problem.set_mesh(mesh_2d, boundary_markers)

init_expr_list = [Expression('x[1]<=25.0 ?' + str(mass_frac_NaOH[0]) + ':' + str(mass_frac_HCl[0]) , degree=1),\
                  Expression('x[1]<=25.0 ?' + str(mass_frac_NaOH[1]) + ':' + str(mass_frac_HCl[1]) , degree=1),\
                  Expression('x[1]<=25.0 ?' + str(mass_frac_NaOH[2]) + ':' + str(mass_frac_HCl[2]) , degree=1),\
                  Expression('x[1]<=25.0 ?' + str(mass_frac_NaOH[3]) + ':' + str(mass_frac_HCl[3]) , degree=1),\
                 ]

dt_num = 0.5
endtime = 300.0

problem.set_transport_species(4, init_expr_list)
problem.set_boundary_conditions()
problem.set_flow_equations(r_num=100.0)
problem.set_transport_equations()

problem.solve(dt_num, endtime)

#problem.output()

#print(rho.vector().local_size())

print('mass_bal_violation = ', problem.mass_bal_violation, 'process = ', MPI.rank(MPI.comm_world))
