from . import *

def points_to_refine(mesh_nd, f_of_phi, threshold = 5):
    # This function returns the points of mesh cells to refine
    # and the phase field dolfin function

    # Inputs
    # mesh_nd: dolfin_generated_mesh
    # f_of_phi: a function returned by scipy.interpolate import RegularGridInterpolator
    # threshold: a threshold value to compare with norm_grad_phi

    # Outputs
    # points:  A list of points to refine. [ [x1, y1], [x2, y2], ... ]
    # phi: dolfin function

    CG_space = FunctionSpace(mesh_nd, 'CG', 1)
    DG_space = FunctionSpace(mesh_nd, 'DG', 0)

    phi = Function(CG_space)

    dof_coordinates = CG_space.tabulate_dof_coordinates()
    dof_coordinates_dg = DG_space.tabulate_dof_coordinates()

    if (mesh_nd.geometric_dimension() == 3):
        dof_x = dof_coordinates[:, 0]
        dof_y = dof_coordinates[:, 1]
        dof_z = dof_coordinates[:, 2]

        dof_x_dg = dof_coordinates_dg[:, 0]
        dof_y_dg = dof_coordinates_dg[:, 1]
        dof_z_dg = dof_coordinates_dg[:, 2]

        #for i in range(len(dof_x)):
        phi.vector()[:] = f_of_phi(np.array([dof_x, dof_y, dof_z]).T)

        norm_grad_phi = project( inner(grad(phi), grad(phi)), DG_space)

        #File("norm_grad_phi.pvd") << norm_grad_phi

        point_index = np.where(norm_grad_phi.vector()>threshold)[0]

        point_x = dof_x_dg[point_index]
        point_y = dof_y_dg[point_index]
        point_z = dof_z_dg[point_index]

        points = np.array([point_x, point_y, point_z]).T

    if (mesh_nd.geometric_dimension() == 2):
        dof_x = dof_coordinates[:, 0]
        dof_y = dof_coordinates[:, 1]

        dof_x_dg = dof_coordinates_dg[:, 0]
        dof_y_dg = dof_coordinates_dg[:, 1]

        #for i in range(len(dof_x)):
        phi.vector()[:] = f_of_phi(np.array([dof_x, dof_y]).T)

        norm_grad_phi = project( inner(grad(phi), grad(phi)), DG_space)

        #File("norm_grad_phi.pvd") << norm_grad_phi

        point_index = np.where(norm_grad_phi.vector()>threshold)[0]

        point_x = dof_x_dg[point_index]
        point_y = dof_y_dg[point_index]

        points = np.array([point_x, point_y]).T

    return points, phi
