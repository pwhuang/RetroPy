#from . import np, rkt
from . import *

def concentration_transport2D(mesh_nd, epsilon, Pe, Da, order, c_left_bc, coordinates):
    # This function solves the steady-state advection diffusion reaction equation of a certain concentration
    # Solves the problem in both Cartesian and Cylindrical coordinates

    # Inputs
    # mesh_2d:        dolfin generated mesh
    # epsilon:        epsilon = Ly/Lx, the aspect ratio of the fracture/cylinder
    # Pe:             The Peclet number
    # Da:             The Damkoehler number
    # order:          order of reaction, currently supports 1 or 2.
    # c_left_bc:      The Dirichlet Boundary condition for the concentration on the left boundary

    # Outputs
    # C:          the velocity field, dolfin function

    #The advection diffusion equation of some concentration
    boundary_markers = MeshFunction('size_t', mesh_nd, mesh_nd.topology().dim() - 1)

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

    ds = Measure('ds', domain=mesh_nd, subdomain_data=boundary_markers)
    dx = Measure('dx', domain=mesh_nd, subdomain_data=boundary_markers)

    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh_nd, P1)

    C = TrialFunction(V)
    v = TestFunction(V)

    eps1 = Constant(epsilon)
    eps2 = Constant(epsilon**2)
    one = Constant(1)
    Pe = Constant(Pe)
    Da = Constant(Da)

    bc0 = DirichletBC(V, c_left_bc, b_left)
    #bc1 = DirichletBC(V, one, right)

    bcs = [bc0]
    #C = split(C)[0]

    #print(c)

    #C = Function(V)
    C = interpolate(one, V)

    f = Constant(0.0)
    if coordinates=='Cartesian':
        u_nd, p_nd = stokes_lubrication(mesh_nd, epsilon**2, top_bottom)

        if order==1:
            a = (C.dx(0)*v.dx(0) + C.dx(1)*v.dx(1)/eps2)*dx + Pe*(u_nd[0]*C.dx(0) + u_nd[1]*C.dx(1))*v*dx\
            - Da/eps2*(1.0-C)*v*(ds(0) + ds(1))
        elif order==2:
            a = (C.dx(0)*v.dx(0) + C.dx(1)*v.dx(1)/eps2)*dx + Pe*(u_nd[0]*C.dx(0) + u_nd[1]*C.dx(1))*v*dx\
            - Da/eps2*(1.0-C*C)*v*(ds(0) + ds(1))
    elif coordinates=='Cylindrical':
        u_nd, p_nd = stokes_lubrication_cylindrical(mesh_nd, epsilon**2, top_bottom)
        r = Expression('x[1]', degree=1)

        if order==1:
            a = (C.dx(0)*v.dx(0) + C.dx(1)*v.dx(1)/eps2)*r*dx + Pe*(u_nd[0]*C.dx(0) + u_nd[1]*C.dx(1))*v*r*dx\
            - Da/eps2*(1.0-C)*v*r*(ds(0))
        elif order==2:
            a = (C.dx(0)*v.dx(0) + C.dx(1)*v.dx(1)/eps2)*dx + Pe*(u_nd[0]*C.dx(0) + u_nd[1]*C.dx(1))*v*dx\
            - Da/eps2*(1.0-C*C)*v*(ds(0))


    solve(a==0, C, bcs, solver_parameters={'newton_solver':{'linear_solver': 'mumps', 'preconditioner': 'default'\
                                                            , 'maximum_iterations': 10,'krylov_solver': {'maximum_iterations': 10000}}})
    return C
