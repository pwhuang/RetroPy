from . import *

def refine_mesh_around_points(mesh_nd, points, depth):
    # Inputs
    # mesh_nd: dolfin_generated_mesh
    # points: A list of points to refine. [ [x1, y1], [x2, y2], ... ]
    # depth: how many times of refinement

    # Outputs:
    # mesh_nd: refined dolfin_generated_mesh
    # report: integer, 1 for successfully performed refinement.
    #                  0 for failure.

    if len(points)==0:
        print('No points to refine with!')
        report = 0
        return mesh_nd, report

    for i in range(depth):
        bbt = mesh_nd.bounding_box_tree()
        collisions = []
        if (mesh_nd.geometric_dimension() == 2):
            for P in points:
                Px = P[0]
                Py = P[1]

                collisions1st = bbt.compute_first_entity_collision(Point(Px, Py))
                collisions.append(collisions1st)

        if (mesh_nd.geometric_dimension() == 3):
            for P in points:
                Px = P[0]
                Py = P[1]
                Pz = P[2]

                collisions1st = bbt.compute_first_entity_collision(Point(Px, Py, Pz))
                collisions.append(collisions1st)

        MF = MeshFunction('bool', mesh_nd, mesh_nd.geometric_dimension())

        #print(np.unique(collisions))
        MF.array()[np.unique(collisions)] = True
        #MF.vector()[np.unique(collisions)] = True
        mesh_nd = refine(mesh_nd, MF)

    report = 1
    return mesh_nd, report
