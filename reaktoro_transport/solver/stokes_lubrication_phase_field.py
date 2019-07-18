from . import *

def stokes_lubrication_phase_field(mesh_2d, eps2, phi):
    # This function solves the Stokes problem defined by the lubrication theory
    # Inputs
    # mesh_2d:        dolfin generated mesh
    # eps2:           epsilon squared, epsilon = Ly/Lx, the aspect ratio of the fracture
    # phi:            the phase field, dolfin function

    # Outputs
    # u_nd:          the velocity field, dolfin function
    # p_nd:          the pressure field, dolfin function

    # Define function spaces
    P2 = VectorElement('Lagrange', triangle, 2)
    P1 = FiniteElement('Lagrange', triangle, 1)

    #TH = P2 * P1
    TH = MixedElement([P2, P1])
    W = FunctionSpace(mesh_2d, TH)
    Vec = VectorFunctionSpace(mesh_2d, 'DG', 0)
    x = SpatialCoordinate(mesh_2d)

    class right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1, DOLFIN_EPS)

    class left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0, DOLFIN_EPS)

    class top_bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[1], 0, DOLFIN_EPS) or near(x[1], 1.0, DOLFIN_EPS))

    def boundary(x, on_boundary):
        return on_boundary

    b_left = left()
    b_right = right()
    b_top_bottom = top_bottom()

    boundary_markers = MeshFunction('size_t', mesh_2d, dim=1)

    b_left.mark(boundary_markers, 1)
    b_right.mark(boundary_markers, 2)
    b_top_bottom.mark(boundary_markers, 3)

    ds = Measure('ds', domain=mesh_2d, subdomain_data=boundary_markers)
    dx = Measure('dx', domain=mesh_2d, subdomain_data=boundary_markers)

    # Boundary condition for pressure at inflow
    one = Constant(1)
    bc0 = DirichletBC(W.sub(1), one, b_left)

    # Boundary condition for pressure at outflow
    zero = Constant(0)
    bc2 = DirichletBC(W.sub(1), zero, b_right)

    # Boundary condition for y-velocity at inflow
    bc3 = DirichletBC(W.sub(0).sub(1), zero, b_left)
    bc4 = DirichletBC(W.sub(0).sub(1), zero, b_right)

    bc_noflow = DirichletBC(W.sub(0), (zero, zero), b_top_bottom)

    # Collect boundary conditions
    bcs = [bc2, bc_noflow]#, bc3, bc4]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    eps2 = Constant(eps2) #interpolate(Expression(eps2, degree=0), W.sub(1).collapse())

    f = Constant((0.0, 0.0))
    delta_t = 1
    Re = 0.1
    dt_x = Constant(delta_t/eps2/Re)
    dt_y = Constant(delta_t/eps2/eps2/Re)
    dt = Constant(delta_t)

    #grad_phi = project(grad(phi), Vec)

    #u_old = interpolate(Constant((0,0)), W.sub(0).collapse())
    hmin = mesh_2d.hmin()
    eta = Constant(1e6) #A large enough number

    F = (u[0].dx(0)*eps2*v[0].dx(0) + u[0].dx(1)*v[0].dx(1) + u[1].dx(0)*v[1].dx(0)*eps2*eps2 + u[1].dx(1)*v[1].dx(1)*eps2)*dx \
        - inner(div(v), p)*dx + q*div(u)*dx - v[0]*ds(1) \
        + eta*phi*inner(u, v)*dx
        #- (grad_phi[0]*eps2*v[0].dx(0) + grad_phi[1]*v[0].dx(1))*u[0]*dx \
        #- (grad_phi[0]*eps2*eps2*v[1].dx(0) + grad_phi[1]*eps2*v[1].dx(1))*u[1]*dx

    # I consider the term "-(one-phi)*v[0]*ds(1)" as P=1 on the left boundary.
    # The last line is the implementation of Nietsche's method, looks cool but I do not know what effect it gives

    a, L = lhs(F), rhs(F)

    U = Function(W)
    solve(a==L, U, bcs,\
    solver_parameters={"linear_solver": "mumps", 'preconditioner': 'default'},\
    form_compiler_parameters={"optimize": False})

    u_nd, p_nd = U.split()

    return u_nd, p_nd
