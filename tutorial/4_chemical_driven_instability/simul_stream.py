import numpy as np
import reaktoro as rkt
from dolfin import *

parameters["ghost_mode"] = "shared_vertex"

import sys
sys.path.insert(0, '../../../Reaktoro-Transport')
from reaktoro_transport.solver import multicomponent_transport_problem

class multicomponent_transport_problem_stream(multicomponent_transport_problem):
    def set_flow_equations(self):
        W = FunctionSpace(self.mesh, 'CG', 1)

        phi = Constant(1.0)
        K = Constant(1.0/48)  # aperture = 0.5 mm
        c = Constant(4e-10)   # Pa^-1
        mu = Constant(8.9e-4)
        g = Constant(-9806.65) # mm^2/s
        zero = Constant(0.0)

        p = TrialFunction(W)
        w = TestFunction(W)
        self.psi = Function(W)

        ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundary_markers)
        dS = Measure('dS', domain=self.mesh, subdomain_data=self.boundary_markers)

        bc_list = [DirichletBC(W, zero, self.boundary_markers, 1),
                   DirichletBC(W, zero, self.boundary_markers, 3),
                   DirichletBC(W, zero, self.boundary_markers, 4),
                   DirichletBC(W, zero, self.boundary_markers, 2)]

        F = inner(grad(w), grad(p))*dx + K/mu*g*w.dx(0)*self.rho*dx

        a, L = lhs(F), rhs(F)

        problem = LinearVariationalProblem(a, L, self.psi, bcs=bc_list)
        self.flow_solver = LinearVariationalSolver(problem)

        prm = self.flow_solver.parameters

        prm['krylov_solver']['absolute_tolerance'] = 1e-14
        prm['krylov_solver']['relative_tolerance'] = 1e-12
        prm['krylov_solver']['maximum_iterations'] = 2000
        prm['linear_solver'] = 'gmres'
        prm['preconditioner'] = 'amg'

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
            #MPI.barrier(MPI.comm_world)
            #self.test_parallel()

            #u0 = potential_flow(self.mesh, self.boundary_markers, self.rho)
            #u_list.append(u0.copy())
            #p_list.append(p0.copy())

            self.flow_solver.solve()

            u0 = project(as_vector([self.psi.dx(1), -self.psi.dx(0)]), self.BDM_space)

            self.adv.assign(u0)
            #rho_list.append(self.rho.copy())
            self.xdmf_obj.write(self.rho, i*dt_num)
            self.xdmf_obj.write(self.adv, i*dt_num)

            for j in range(self.num_transport_components):
                self.solver_list[j].solve()

                self.X_list_old[j].assign(self.X_list[j])
                self.xdmf_obj.write(self.X_list[j], i*dt_num)

                #out_list[j].append(self.X_list_old[j].copy())

            #MPI.barrier(MPI.comm_world)
            sum_violation = int(MPI.sum(MPI.comm_world, self.mass_bal_violation))
            begin('violation count = ' + str(sum_violation))
            if sum_violation == 0:
                dt_num = dt_num*1.0
                self.dt.assign(dolfin.Constant(dt_num))
            elif sum_violation > 0:
                dt_num = dt_num*0.75
                self.dt.assign(dolfin.Constant(dt_num))
            end()


        self.xdmf_obj.close()

        return out_list, u_list, p_list

    def output(self):
        xdmf_obj = dolfin.XDMFFile(MPI.comm_world, 'solution_output.xdmf')
        xdmf_obj.write(self.par_test, 0)
        xdmf_obj.close()



mesh_2d = RectangleMesh.create(MPI.comm_world, [Point(0.0, 0.0), Point(31.0, 50.0)], [30, 50], CellType.Type.triangle, 'right/left')
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

problem = multicomponent_transport_problem('solution_output.xdmf')

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

problem.set_transport_species(4, init_expr_list)
problem.set_boundary_conditions()
problem.set_flow_equations()
problem.set_transport_equations()

dt_num = 0.5
timesteps = 2
u_list_mult, u_list, p_list = problem.solve(dt_num, timesteps)

#problem.output()

#print(rho.vector().local_size())

print('mass_bal_violation = ', problem.mass_bal_violation, 'process = ', MPI.rank(MPI.comm_world))
