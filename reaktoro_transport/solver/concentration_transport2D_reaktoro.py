#from . import np, rkt
from . import *

def concentration_transport2D_reaktoro(mesh_2d, epsilon, Pe_num\
                                       , c_left_bc, init_expr, t_scale\
                                       , dt_num, time_steps, theta_num, reaktoro_init):
    # This function solves the transient advection diffusion reaction equation of a certain concentration
    # using theta family of methods.
    # Solves the problem in the Cartesian coordinates

    # Inputs
    # mesh_2d:        dolfin generated mesh
    # epsilon:        epsilon = Ly/Lx, the aspect ratio of the fracture
    # Pe_num:         The Peclet number
    # c_left_bc:      The Dirichlet Boundary condition for the concentration on the left boundary
    # init_expr:      The initial condition of C. Defined using the Expression function.
    # dt_num:         The nondimensional delta t
    # time_steps:     Integer. How many steps of time marching
    # theta_num:      theta=1: implict scheme, theta=0: explicit scheme, theta=0.5: the Crank-Nicolson scheme
    # reaktoro_init:  ... explain in the tutorial

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

    bc0 = DirichletBC(V, c_left_bc, b_left)
    #bc1 = DirichletBC(V, one, right)

    bcs = [bc0]

    #The boundary conditions set here is to get boundary values
    bc_top = DirichletBC(V, 1, b_top)
    bc_bottom = DirichletBC(V, 2, b_bottom)
    boundary_indicator = Function(V) #Dummy Function

    bc_top.apply(boundary_indicator.vector())
    bc_bottom.apply(boundary_indicator.vector())

    C_old = interpolate(init_expr, V)

    u_nd, p_nd = stokes_lubrication(mesh_2d, epsilon**2, top_bottom)

    dt = Constant(dt_num)
    theta = Constant(theta_num) # 1 = implicit scheme

    C_list = [C_old.copy()]

    theta = Constant(theta_num)

    # Use the reaktoro_init function to generate the states that reaktoro_solve_rates function
    state0, path, reactions, C_scale = reaktoro_init()

    # initiate a reaktoro_flux function
    reactoro_flux = interpolate(Expression('0', degree=1), V)

    for i in range(time_steps):
        reactoro_flux.vector()[np.where(boundary_indicator.vector()[:]==1)[0]] = tools.reaktoro_solve_rates(C_old.vector()[boundary_indicator.vector()[:] == 1], C_scale, t_scale, reactions, state0)
        reactoro_flux.vector()[np.where(boundary_indicator.vector()[:]==2)[0]] = tools.reaktoro_solve_rates(C_old.vector()[boundary_indicator.vector()[:] == 2], C_scale, t_scale, reactions, state0)

        # This should be rewritten to avoid assembly of system of equations
        C = TrialFunction(V)

        F = (C-C_old)/dt*v*dx \
        + theta*(C.dx(0)*v.dx(0) + C.dx(1)*v.dx(1)/eps2)*dx \
        + theta*Pe*(u_nd[0]*C.dx(0) + u_nd[1]*C.dx(1))*v*dx \
        - reactoro_flux*v*(ds(0) + ds(1)) \
        + (one-theta)*(C_old.dx(0)*v.dx(0) + C_old.dx(1)*v.dx(1)/eps2)*dx \
        + (one-theta)*Pe*(u_nd[0]*C_old.dx(0) + u_nd[1]*C_old.dx(1))*v*dx
        #- (one-theta)*Constant(reactoro_flux)*v*(ds(0) + ds(1))

        a, L = lhs(F), rhs(F)

        C = Function(V)

        solve(a==L, C, bcs, solver_parameters={'linear_solver': 'gmres', 'preconditioner': 'ilu'})

        C_old.assign(C)

        C_list.append(C_old.copy())

    return C_list
