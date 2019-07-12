#from . import np, rkt
from . import *

def stokes_lubrication(mesh_2d, eps2, top_bottom):
    # This function solves the Stokes problem defined by the lubrication theory
    # Inputs
    # mesh_2d:        dolfin generated mesh
    # eps2:           epsilon squared, epsilon = Ly/Lx, the aspect ratio of the fracture
    # top_bottom:     A subdomain class that defines the top and bottom boundary of the fracture

    # Outputs
    # u_nd:          the velocity field, dolfin function
    # p_nd:          the pressure field, dolfin function

    # Define function spaces
    P2 = VectorElement('Lagrange', triangle, 1) #VectorElement("P",mesh.ufl_cell(),2)
    P1 = FiniteElement('Lagrange', triangle, 1) #FiniteElement("P",mesh.ufl_cell(),1)
    #TH = P2 * P1
    TH = MixedElement([P2, P1])
    W = FunctionSpace(mesh_2d,TH)
    x = SpatialCoordinate(mesh_2d)

    b_top_bottom = top_bottom()

    class right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1, DOLFIN_EPS)

    class left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary(x, on_boundary):
        return on_boundary

    b_left = left()
    b_right = right()

    boundary_markers = MeshFunction('size_t', mesh_2d, dim=1)

    b_left.mark(boundary_markers, 1)
    b_right.mark(boundary_markers, 2)
    b_top_bottom.mark(boundary_markers, 0)

    ds = Measure('ds', domain=mesh_2d, subdomain_data=boundary_markers)
    dx = Measure('dx', domain=mesh_2d, subdomain_data=boundary_markers)

    # No-slip boundary condition for velocity
    noslip = Constant((0.0, 0.0))
    bc1 = DirichletBC(W.sub(0), noslip, b_top_bottom)

    # Boundary condition for pressure at inflow
    one = Constant(1)
    bc0 = DirichletBC(W.sub(1), one, b_left)

    # Boundary condition for pressure at outflow
    zero = Constant(0)
    bc2 = DirichletBC(W.sub(1), zero, b_right)

    # Boundary condition for y-velocity at inflow
    bc3 = DirichletBC(W.sub(0).sub(1), zero, b_left)
    bc4 = DirichletBC(W.sub(0).sub(1), zero, b_right)

    # Collect boundary conditions
    bcs = [bc0, bc2, bc1]#, bc3, bc4]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    eps2 = Constant(eps2) #interpolate(Expression(eps2, degree=0), W.sub(1).collapse())

    f = Constant((0.0, 0.0))

    F = -(u[0].dx(0)*v[0].dx(0)*eps2 + u[0].dx(1)*v[0].dx(1) + u[1].dx(0)*v[1].dx(0)*eps2*eps2 + u[1].dx(1)*v[1].dx(1)*eps2)*dx \
        + (-inner(v, grad(p)) + q*div(u))*dx
        #+ 2.0*v[0]*deps2_dx*u[0].dx(0)*dx

    a, L = lhs(F), rhs(F)

    U = Function(W)
    solve(a==L, U, bcs)
    #solve(F==0, U.vector(), bcs)

    u_nd, p_nd = U.split()

    return u_nd, p_nd
