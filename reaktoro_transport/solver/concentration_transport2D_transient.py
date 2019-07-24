#from . import np, rkt
from . import *

def concentration_transport2D_transient(mesh_2d, epsilon, Pe_num, Da_num\
                                       , c_left_bc, init_expr\
                                       , dt_num, time_steps, theta_num):
    # This function solves the transient advection diffusion reaction equation of a certain concentration
    # using theta family of methods.
    # Solves the problem in the Cartesian coordinates

    # Inputs
    # mesh_2d:        dolfin generated mesh
    # epsilon:        epsilon = Ly/Lx, the aspect ratio of the fracture
    # Pe_num:         The Peclet number
    # Da_num:         The Damkoehler number
    # c_left_bc:      The Dirichlet Boundary condition for the concentration on the left boundary
    # init_expr:      The initial condition of C. Defined using the Expression function.
    # dt_num:         The nondimensional delta t
    # time_steps:     Integer. How many time_steps of time marching
    # theta_num:      theta=1: implict scheme, theta=0: explicit scheme, theta=0.5: the Crank-Nicolson scheme

    # Outputs
    # C:          the concentration field, dolfin function

    boundary_markers = MeshFunction('size_t', mesh_2d, mesh_2d.topology().dim() - 1)

    class top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 1, DOLFIN_EPS)# and (x[0] > DOLFIN_EPS)

    class bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0, DOLFIN_EPS)# and (x[0] > DOLFIN_EPS)

    class right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1, DOLFIN_EPS)

    class left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0, DOLFIN_EPS)# and (x[1] > DOLFIN_EPS) and (x[1] < 1-DOLFIN_EPS)

    class top_bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[1], 0, DOLFIN_EPS) or near(x[1], 1.0, DOLFIN_EPS))

    b_top = top()
    b_bottom = bottom()
    b_right = right()
    b_left = left()

    b_top.mark(boundary_markers, 0)
    b_bottom.mark(boundary_markers, 1)
    b_right.mark(boundary_markers, 2)
    b_left.mark(boundary_markers, 3)

    ds = Measure('ds', domain=mesh_2d, subdomain_data=boundary_markers)
    dx = Measure('dx', domain=mesh_2d, subdomain_data=boundary_markers)

    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh_2d, P1)

    C = TrialFunction(V)
    v = TestFunction(V)

    eps1 = Constant(epsilon)
    eps2 = Constant(epsilon**2)
    one = Constant(1)
    Pe = Constant(Pe_num)
    Da = Constant(Da_num)

    bc0 = DirichletBC(V, c_left_bc, b_left)
    #bc1 = DirichletBC(V, one, right)

    bcs = [bc0]

    #The boundary conditions set here is to get boundary values
    bc_top = DirichletBC(V, 1, b_top)
    bc_bottom = DirichletBC(V, 2, b_bottom)
    u = Function(V) #Dummy Function

    bc_top.apply(u.vector())
    bc_bottom.apply(u.vector())

    C_old = interpolate(init_expr, V)

    u_nd, p_nd = stokes_lubrication(mesh_2d, epsilon**2, top_bottom)

    dt = Constant(dt_num)
    theta = Constant(theta_num) # 1 = implicit scheme

    C_list = [C_old.copy()]

    F = (C-C_old)/dt*v*dx \
    + theta*((C.dx(0)*v.dx(0) + C.dx(1)*v.dx(1)/eps2)*dx \
    + Pe*(u_nd[0]*C.dx(0) + u_nd[1]*C.dx(1))*v*dx \
    - Da/eps2*(1.0-C)*v*(ds(0) + ds(1))) \
    + (one-theta)*((C_old.dx(0)*v.dx(0) + C_old.dx(1)*v.dx(1)/eps2)*dx\
    + Pe*(u_nd[0]*C_old.dx(0) + u_nd[1]*C_old.dx(1))*v*dx \
    - Da/eps2*(1.0-C_old)*v*(ds(0) + ds(1)))

    a, L = lhs(F), rhs(F)
    C = Function(V)

    for i in range(time_steps):
        solve(a==L, C, bcs, solver_parameters={'linear_solver': 'gmres',\
                         'preconditioner': 'ilu'})
        #, solver_parameters={'newton_solver':{'linear_solver': 'mumps', 'preconditioner': 'default'\
        #                                        , 'maximum_iterations': 10,'krylov_solver': {'maximum_iterations': 10000}}})
        C_old.assign(C)
        # Add the solution to the list
        C_list.append(C_old.copy())

    return C_list
