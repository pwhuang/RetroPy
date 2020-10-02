from . import *

def stokes_uzawa(mesh, boundary_markers, boundary_dict\
                 , p_list=[1.0], init_p=Constant(0.0)\
                 , max_steps=500, res_target=1e-12, omega_num=1.0, r_num=0.0):
    # The Augmented Lagrangian method is implemented.
    # When r_num=0, converges for omega_num < 2. (Try 1.5 first)
    # For 0 < omega < 2r, the augmented system converges. r>>1

    V = VectorFunctionSpace(mesh, "Crouzeix-Raviart", 1)
    CG1 = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    dS = Measure('dS', domain=mesh, subdomain_data=boundary_markers)

    bcu = []
    bcp = []

    if mesh.geometric_dimension()==2:
        noslip = (0.0, 0.0)
    elif mesh.geometric_dimension()==3:
        noslip = (0.0, 0.0, 0.0)

    for idx in boundary_dict['noslip']:
        bcu.append(DirichletBC(V, noslip, boundary_markers, idx))

    # Create functions
    u0 = Function(V)
    u1 = Function(V)
    v0 = Function(V) # Test function used for residual calculations
    for bc in bcu:
        bc.apply(v0.vector())

    #p0 = project(init_p, Q, solver_type='gmres', preconditioner_type='amg')
    p0 = interpolate(init_p, Q)
    p1 = Function(Q)

    # Define coefficients
    f = Constant((0, 0))
    beta = Constant(1e1)
    one = Constant(1.0)
    r = Constant(r_num)
    omega = Constant(omega_num)
    h = CellDiameter(mesh)

    nn = FacetNormal(mesh)
    h_avg = (h('+') + h('-'))/2.0

    def precond_u():
        return inner(grad(v), grad(u-u0))*dx

    # Tentative velocity step
    F1 = precond_u() \
         + inner(grad(v), grad(u0))*dx \
         - inner(p0, div(v))*dx \
         + r*inner(div(v), div(u))*dx

    for i, p_dirichlet in enumerate(p_list):
         F1 += Constant(p_dirichlet)*inner(nn, v)*ds(boundary_dict['inlet'][i])\
               - dot(nn, dot(grad(u), v))*ds(boundary_dict['inlet'][i])

    a1 = lhs(F1)
    L1 = rhs(F1)

    # Assemble matrices
    A1 = assemble(a1)

    # Pressure update
    #a2 = q*p*dx  + omega*(2.0 - (phi('+') + phi('-')))*h_avg**2*dot(jump(q), jump(p))*dS
    a2 = q*p*dx #+ omega*h_avg*dot(jump(q), jump(p))*dS(0) #beta = 1.0
    L2 = q*p0*dx - omega*q*div(u1)*dx

#     a2 = q*p*dx
#     L2 = q*p0*dx - omega*q*div(u1)*dx + omega*h_avg*dot(jump(q), jump(p0))*dS

#     a2 = omega*h_avg*dot(jump(q), jump(p))*dS #beta = 1.0
#     L2 = phi_DG*omega*inner(grad(q), u1)*dx

    A2 = assemble(a2)

    u_list = []
    p_list = []
    res_list = []

    #solver1 = PETScKrylovSolver('gmres', 'hypre_amg')
    solver1 = PETScLUSolver('mumps')
    solver2 = PETScKrylovSolver('gmres', 'amg')

    prm = solver1.parameters

    #prm['absolute_tolerance'] = 1e-14
    #prm['ksp_converged_reason'] = True
    #prm['relative_tolerance'] = 1e-12
    #prm['maximum_iterations'] = 20000
    #prm['error_on_nonconvergence'] = True
    #prm['monitor_convergence'] = True
    #prm['nonzero_initial_guess'] = True

    prm = solver2.parameters

    prm['absolute_tolerance'] = 1e-14
    #prm['ksp_converged_reason'] = True
    prm['relative_tolerance'] = 1e-12
    prm['maximum_iterations'] = 2000
    prm['error_on_nonconvergence'] = True
    prm['monitor_convergence'] = True
    prm['nonzero_initial_guess'] = True

    #pout = File("Box_p_iter.pvd")
    #vout = File("Box_v_iter.pvd")

    xdmf_obj = XDMFFile(MPI.comm_world, 'pv_output.xdmf')

    residual = 1.0
    div_u = 666
    p_diff = 111
    i = 0

    #for i in range(steps):
    while(np.abs(residual) > res_target):
        if (i>=max_steps):
            if MPI.rank(MPI.comm_world)==0:
                print('Reached maximum steps! Saving progress...')
            break

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solver1.solve(A1, u1.vector(), b1)
        end()

        # Pressure correction
        #begin("Computing pressure correction")
        b2 = assemble(L2)
        #[bc.apply(A2, b2) for bc in bcp]
        solver2.solve(A2, p1.vector(), b2)
        #end()

        #div_u1 = project(div(u1), Q)
        #p1.vector()[:] = p0.vector()[:] - omega_num*div_u1.vector()[:]

        p_diff = assemble((p1-p0)**2*dx)**0.5
        div_u = assemble(q*div(u1)*dx).norm('l2')
        residual_form = (inner(grad(v0), grad(u1)) - p1*div(v0))*dx
        for i, p_dirichlet in enumerate(p_list):
             residual_form += Constant(p_dirichlet)*inner(nn, v0)*ds(boundary_dict['inlet'][i])\
                              - dot(nn, dot(grad(u), v0))*ds(boundary_dict['inlet'][i])
        residual = assemble(residual_form) + div_u

        u0.assign(u1)
        p0.assign(p1)

        if MPI.rank(MPI.comm_world)==0:
            #print(div_u, p_diff)
            res_list.append(residual)

        i+=1

        begin('Step ' + str(i) + ', residual = ' + str(residual))
        end()

    # Only saving the last time step
    uCG = project(u0, CG1, solver_type='gmres', preconditioner_type='amg')
    xdmf_obj.write_checkpoint(uCG, 'velocity_CG1', 0, append=False)
    xdmf_obj.write_checkpoint(u0, 'velocity', 0, append=True)
    xdmf_obj.write_checkpoint(p0, 'pressure', 0, append=True)

    xdmf_obj.close()

    if MPI.rank(MPI.comm_world)==0:
        #u_list.append(u0.copy())
        #p_list.append(p0.copy())
        print(div_u, p_diff)
        print('Used  ', i, ' steps to converge!')

    return u0.copy(), p0.copy(), res_list
