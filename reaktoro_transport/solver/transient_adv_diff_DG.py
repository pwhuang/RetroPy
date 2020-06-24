from . import *
from ufl.algebra import Abs

def transient_adv_diff_DG(mesh, boundary_markers, adv, source, D_num, init_cond, dt_num, steps, theta_num):
    # Input
    # mesh:     dolfin mesh

    # Output
    # u_list:   list of dolfin function

    V = FunctionSpace(mesh, 'DG', 0)
    Vec = VectorFunctionSpace(mesh, 'CR', 1)

    u = TrialFunction(V)
    w = TestFunction(V)
    u0 = project(init_cond, V)

    n = FacetNormal(mesh)

    x_ = interpolate(Expression("x[0]", degree=1), V)
    y_ = interpolate(Expression("x[1]",degree=1), V)

    Delta_h = sqrt(jump(x_)**2 + jump(y_)**2)
    adv = project(adv, Vec)

    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    dS = Measure('dS', domain=mesh, subdomain_data=boundary_markers)

    u_list = [u0.copy()]

    bc = []

    dt = Constant(dt_num)
    one = Constant(1.0)

    adv_np = ( dot ( adv, n ) + Abs ( dot ( adv, n ) ) ) / 2.0
    adv_nm = ( dot ( adv, n ) - Abs ( dot ( adv, n ) ) ) / 2.0
    adv_n = dot ( adv, n )

    def L(w, u):
        return Constant(D_num)*dot(jump(w, n), jump(u, n))/Delta_h*dS(0) \
               + dot(jump(w), adv_np('+')*u('+') + adv_nm('+')*u('-') )*dS(0) \
               + w*u/x_*ds(2)

        #return dot(jump(w), adv_np('+')*u('+') - adv_np('-')*u('-') )
        #return Constant(0.5)*w*div(adv*u) + Constant(0.5)*w*dot(adv, grad(u))


    delta_u = u - u0
    W = 0.5
    #ww = (w*source-L(w, u0))

    #F = (w*delta_u/dt + W*L(w, u - u0) - ww)*dx

    #a, L = lhs(F), rhs(F)

    theta = Constant(theta_num)

    a = ( w*u/dt )*dx + theta*L(w, u)
    L = ( w*u0/dt + w*source )*dx - (one-theta)*L(w, u0) + w*one/x_*ds(2)
    u = Function(V)

    problem = LinearVariationalProblem(a, L, u, bcs=bc)
    solver = LinearVariationalSolver(problem)

    prm = solver.parameters

    prm['krylov_solver']['absolute_tolerance'] = 1e-12
    prm['krylov_solver']['relative_tolerance'] = 1e-10
    prm['krylov_solver']['maximum_iterations'] = 500
    #if iterative_solver:
    prm['linear_solver'] = 'gmres'
    prm['preconditioner'] = 'ilu'

    for i in range(steps):
        solver.solve()
        u0.assign(u)
        u_list.append(u0.copy())

    return u_list
