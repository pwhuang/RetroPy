#from . import np, rkt
from . import *

def concentration_transport_phase_field(mesh_2d, epsilon, Pe, Da, c_left_bc, u_nd, phi):
    # This function solves the steady-state advection diffusion reaction equation of a certain concentration

    # Inputs
    # mesh_2d:        dolfin generated mesh
    # epsilon:        epsilon = Ly/Lx, the aspect ratio of the fracture
    # Pe:             The Peclet number
    # Da:             The Damkoehler number
    # c_left_bc:      The Dirichlet Boundary condition for the concentration on the left boundary
    # u_nd:           The velocity field solved by stokes_lubrication_phase_field
    # phi:            The phase field, first order CG function

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
            return on_boundary and near(x[0], 0, DOLFIN_EPS)# and (x[1] > 0.5-DOLFIN_EPS) and (x[1] < 1.5+DOLFIN_EPS)

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
    Vec = VectorFunctionSpace(mesh_2d, 'DG', 0)
    DG_space = FunctionSpace(mesh_2d, 'DG', 0)

    C = TrialFunction(V)
    v = TestFunction(V)

    eps1 = Constant(epsilon)
    eps2 = Constant(epsilon**2)
    one = Constant(1)
    eta = Constant(1e6)
    Pe = Constant(Pe)
    Da = Constant(Da)

    bc0 = DirichletBC(V, c_left_bc, b_left)

    bcs = [bc0]
    C = interpolate(one, V)

    f = Constant(0.0)

    grad_phi = project(grad(phi), Vec)
    norm_grad_phi = project(sqrt( eps2*grad_phi[0]*grad_phi[0] + grad_phi[1]*grad_phi[1] ), DG_space)

    a = (1-phi)*(C.dx(0)*v.dx(0) + C.dx(1)*v.dx(1)/eps2)*dx + Pe*(u_nd[0]*C.dx(0) + u_nd[1]*C.dx(1))*v*dx\
    - norm_grad_phi*Da/eps2*(1.0-C)*v*dx #+ inner(grad_phi, grad(v))*C*dx #Nietche's method

    solve(a==0, C, bcs, solver_parameters={'newton_solver':{'linear_solver': 'mumps', 'preconditioner': 'default'\
                                                         , 'maximum_iterations': 10,'krylov_solver': {'maximum_iterations': 10000}}})
    return C
