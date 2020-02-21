from . import *

def stokes_uzawa2D(mesh, phi, boundary_markers, steps, omega_num=1.0, r_num=0.0, pressure_scale=1.0):
    # This code is still under development!
    # phi should be defined in DG0!
    # The Augmented Lagrangian method is implemented.
    # When r_num=0, converges for omega_num < 2. (Try 1.5 first)
    # For 0 < omega < 2r, the augmented system converges. r>>1

    P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)

    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    Q = FunctionSpace(mesh, "DG", 0)

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)


    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    dS = Measure('dS', domain=mesh, subdomain_data=boundary_markers)

    noslip  = DirichletBC(V, (0.0, 0.0), boundary_markers, 21)
    noslip_interior = DirichletBC(V, (0.0, 0.0), boundary_markers, 20)

    bcu = [noslip, noslip_interior]
    bcp = []

    # Create functions
    u0 = Function(V)
    u1 = Function(V)

    p0 = Function(Q)
    p1 = Function(Q)

    p0 = project(Expression('x[1]', degree=1), Q)

    # Define coefficients
    f = Constant((0, 0))
    eta = Constant(1e6)
    one = Constant(1.0)
    r = Constant(r_num)
    omega = Constant(omega_num)
    p_scale = Constant(pressure_scale)
    h = CellDiameter(mesh)

    nn = FacetNormal(mesh)
    h_avg = (h('+') + h('-'))/2.0

    def precond_u():
        return inner(grad(v), grad(u-u0))*dx

    # Tentative velocity step
    # -v[1] means the velocity is negative (going down!), this term can be viewed as a dp/dx term.
    F1 = precond_u() \
         - p_scale*(one-phi)*(-v[1])*ds(10) \
         + inner(grad(v), grad(u0))*dx \
         - inner(p0, div(v))*dx \
         + r*inner(div(v), div(u))*dx

    a1 = lhs(F1)
    L1 = rhs(F1)

    # Assemble matrices
    A1 = assemble(a1)

    # Pressure update with a jump penalty
    #a2 = q*p*dx  + omega*(2.0 - (phi('+') + phi('-')))*h_avg**2*dot(jump(q), jump(p))*dS
    a2 = q*p*dx  + omega*h_avg**2*dot(jump(q), jump(p))*dS(0)
    L2 = q*p0*dx - omega*q*div(u1)*dx

    A2 = assemble(a2)

    u_list = []
    p_list = []
    div_u_list = []

    solver1 = PETScKrylovSolver('gmres', 'sor')
    solver2 = PETScKrylovSolver('gmres', 'sor')

    prm = solver1.parameters

    prm['absolute_tolerance'] = 1e-12
    #prm['ksp_converged_reason'] = True
    prm['relative_tolerance'] = 1e-10
    prm['maximum_iterations'] = 2000
    prm['error_on_nonconvergence'] = True
    prm['monitor_convergence'] = True
    prm['nonzero_initial_guess'] = True

    prm = solver2.parameters

    prm['absolute_tolerance'] = 1e-12
    #prm['ksp_converged_reason'] = True
    prm['relative_tolerance'] = 1e-10
    prm['maximum_iterations'] = 2000
    prm['error_on_nonconvergence'] = True
    prm['monitor_convergence'] = True
    prm['nonzero_initial_guess'] = True

    #pout = File("Box_p_iter.pvd")
    #vout = File("Box_v_iter.pvd")

    xdmf_obj = XDMFFile(MPI.comm_world, 'pv_output.xdmf')

    div_u = 1.0
    i = 0

    #for i in range(steps):
    while(np.abs(div_u) > 1e-8):
        # Compute tentative velocity step
        #begin("Computing tentative velocity")
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solver1.solve(A1, u1.vector(), b1)
        #end()

        # Pressure correction
        #begin("Computing pressure correction")
        b2 = assemble(L2)
        #[bc.apply(A2, b2) for bc in bcp]
        solver2.solve(A2, p1.vector(), b2)
        #end()

        p_diff = assemble((p1-p0)**2*dx)
        div_u = assemble((div(u1)*dx))

        u0.assign(u1)
        p0.assign(p1)

        if MPI.rank(MPI.comm_world)==0:
            print(div_u, p_diff)
            div_u_list.append(div_u)

        i+=1
        xdmf_obj.write(u0, i)
        xdmf_obj.write(p0, i)

        if (i>=steps):
            if MPI.rank(MPI.comm_world)==0:
                print('Reached maximum steps! Saving progress...')
            break

    # Only saving the last time step
    #pout << (p0, i)
    #vout << (u0, i)
    xdmf_obj.close()

    if MPI.rank(MPI.comm_world)==0:
        u_list.append(u0.copy())
        p_list.append(p0.copy())
        print('Used  ', i, ' steps to converge!')

    return u_list, p_list, div_u_list
