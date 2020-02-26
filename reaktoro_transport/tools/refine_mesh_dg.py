from . import *

def refine_mesh_dg(mesh_nd, f_of_phi, threshold=1e-4, min_cell_size=1.0):
    # Inputs
    # mesh_nd:   dolfin_generated_mesh
    # f_of_phi:  a scipy RegularGridInterpolator using nearest interpolation
    # threshold: a threshold value for finding the f/s boundaries
    # min_cell_size: the minimum_cell_size of the refined mesh

    # Outputs:
    # mesh_nd: refined dolfin_generated_mesh
    # phi_DG: unfiltered phi_DG

    iteration = 0
    report = 1
    while(report==1):
        CG_space = FunctionSpace(mesh_nd, 'CG', 1)
        DG_space = FunctionSpace(mesh_nd, 'DG', 0)

        phi = Function(CG_space)
        cell_size = project(CellDiameter(mesh_nd), DG_space)

        dof_coordinates_cg = CG_space.tabulate_dof_coordinates()

        if (mesh_nd.geometric_dimension() == 2):
            dof_x = dof_coordinates_cg[:, 0]
            dof_y = dof_coordinates_cg[:, 1]

            phi.vector()[:] = f_of_phi(np.array([dof_x, dof_y]).T)

        if (mesh_nd.geometric_dimension() == 3):
            dof_x = dof_coordinates_cg[:, 0]
            dof_y = dof_coordinates_cg[:, 1]
            dof_z = dof_coordinates_cg[:, 2]

            phi.vector()[:] = f_of_phi(np.array([dof_x, dof_y, dof_z]).T)

        # Need to elaborate and understand more on this concept
        phi_DG = project(phi, DG_space)
        phi_DG_diff = Function(DG_space)

        # This tells you where to refine
        for i, phi_val in enumerate(phi_DG.vector()[:]):
            if phi_val > threshold and phi_val < (1.0 - threshold) and cell_size.vector()[i] > min_cell_size:
                phi_DG_diff.vector()[i] = 1 # True
            else:
                phi_DG_diff.vector()[i] = 0 # False

        #plt.figure()
        #plot(phi_DG_diff)
        #plt.show()

        if np.sum(phi_DG_diff.vector()[:])==0:
            print('Used ', iteration, 'iterations. Mesh generation successful!')
            report = 0
            break

        # Change the function into a MeshFunction
        cell_markers = MeshFunction('bool', mesh_nd, dim=mesh_nd.geometric_dimension())
        cell_markers.array()[:] = phi_DG_diff.vector()[:]

        mesh_nd = refine(mesh_nd, cell_markers)
        iteration = iteration + 1

    return mesh_nd, phi_DG
