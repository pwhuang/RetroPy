from . import *

def refine_mesh_around_points(mesh_nd, points, depth):
    # Inputs
    # mesh_nd: dolfin_generated_mesh
    # points: A list of points to refine. [ [x1, y1], [x2, y2], ... ]
    # depth: how many times of refinement

    for i in range(depth):
        bbt = mesh_nd.bounding_box_tree()
        collisions = []
        for P in points:
            Px = P[0]
            Py = P[1]

            collisions1st = bbt.compute_first_entity_collision(Point(Px, Py))
            collisions.append(collisions1st)

        MF = MeshFunction('bool', mesh_nd, mesh_nd.geometric_dimension())

        #print(np.unique(collisions))
        MF.array()[np.unique(collisions)] = True
        #MF.vector()[np.unique(collisions)] = True
        mesh_nd = refine(mesh_nd, MF)

    return mesh_nd
