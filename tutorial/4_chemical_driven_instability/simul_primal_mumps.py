import numpy as np
import reaktoro as rkt
from dolfin import *
import dolfin

parameters["ghost_mode"] = "shared_vertex"

import sys
sys.path.insert(0, '../../../Reaktoro-Transport')
from reaktoro_transport.solver import multicomponent_transport_problem

class multicomponent_transport_problem_mumps(multicomponent_transport_problem):
    def set_flow_equations(self):
        RT = dolfin.FiniteElement('BDM', self.mesh.cell_name(), 1)
        #RT = VectorElement('CR', mesh.cell_name(), 1)
        DG = dolfin.FiniteElement('DG', self.mesh.cell_name(), 0)

        DG_space = dolfin.FunctionSpace(self.mesh, 'DG', 0)

        self.p_ref = 101450  #Pa
        self.p0 = project(Expression('9.81*(25.0-x[1]) + pref', degree=1, pref = self.p_ref), DG_space)

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

        self.drho_dt = (self.rho - self.rho_old)/self.dt

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

problem = multicomponent_transport_problem_mumps('solution_output_primal_mumps.xdmf')

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

dt_num = 0.5
dt_end = 30.0

problem.set_transport_species(4, init_expr_list)
problem.set_boundary_conditions()
problem.set_flow_equations()
problem.set_transport_equations()

u_list_mult, u_list, p_list = problem.solve(dt_num, dt_end)

#problem.output()

#print(rho.vector().local_size())

print('mass_bal_violation = ', problem.mass_bal_violation, 'process = ', MPI.rank(MPI.comm_world))
