#from . import np, rkt
from . import *

def stokes_lubrication_cylindrical(mesh_2d, eps2, top):
    # This function solves the Stokes problem defined by the lubrication theory in the cylindrical coordinates
    # Inputs
    # mesh_2d:        dolfin generated mesh
    # eps2:           epsilon squared, epsilon = R/Lz, the aspect ratio of the cylinder
    # top:            A subdomain class that defines the top boundary of the cylinder

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

    b_top = top()

    class right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1, DOLFIN_EPS)

    class left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0, DOLFIN_EPS)

    class bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0, DOLFIN_EPS)

    def boundary(x, on_boundary):
        return on_boundary

    b_left = left()
    b_right = right()
    b_bottom = bottom()

    boundary_markers = MeshFunction('size_t', mesh_2d, dim=1)

    b_left.mark(boundary_markers, 1)
    b_right.mark(boundary_markers, 2)
    b_bottom.mark(boundary_markers, 3)
    b_top.mark(boundary_markers, 0)

    ds = Measure('ds', domain=mesh_2d, subdomain_data=boundary_markers)
    dx = Measure('dx', domain=mesh_2d, subdomain_data=boundary_markers)

    # No-slip boundary condition at the top boundary
    noslip = Constant((0.0, 0.0))
    bc1 = DirichletBC(W.sub(0), noslip, b_top)

    # No-flow boundary condition at the bottom boundary
    zero = Constant(0)
    bc5 = DirichletBC(W.sub(0).sub(1), zero, b_bottom)

    # Boundary condition for pressure at inflow
    one = Constant(1)
    bc0 = DirichletBC(W.sub(1), one, b_left)

    # Boundary condition for pressure at outflow
    bc2 = DirichletBC(W.sub(1), zero, b_right)

    # Boundary condition for the r-velocity at inflow
    bc3 = DirichletBC(W.sub(0).sub(1), zero, b_left)
    bc4 = DirichletBC(W.sub(0).sub(1), zero, b_right)

    # Collect boundary conditions
    bcs = [bc0, bc2, bc1, bc5]#, bc3, bc4]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    eps2 = Constant(eps2)
    r = Expression('x[1]', degree=1)

    f = Constant((0.0, 0.0))

    #u[0]: u_r
    #u[1]: u_z

    F = -(u[1].dx(1)*v[1].dx(1)*eps2 + u[1].dx(0)*v[1].dx(0)*eps2*eps2 \
        + u[0].dx(1)*v[0].dx(1) + u[0].dx(0)*v[0].dx(0)*eps2)*r*dx \
        -(v[1]*(p.dx(1) + eps2*u[1]/r/r) + v[0]*p.dx(0))*r*dx \
        + q*(u[1]/r + u[1].dx(1) + u[0].dx(0))*r*dx

    a, L = lhs(F), rhs(F)

    U = Function(W)
    solve(a==L, U, bcs)

    u_nd, p_nd = U.split()

    return u_nd, p_nd
