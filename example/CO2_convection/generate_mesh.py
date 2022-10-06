# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import pygmsh, meshio
from optimesh import optimize_points_cells
import numpy as np

import matplotlib.pyplot as plt

filepath = 'mesh.xdmf'

box_width = 165.0 # mm
box_height = 210.0

segments_count = 6
segments = np.linspace(0.0, box_height, segments_count)

polygon = [[0.0, 0.0]]

for segment in segments:
    polygon.append([box_width, segment])

for segment in np.flip(segments)[:-1]:
    polygon.append([0.0, segment])

y_vals = np.append(np.array(polygon)[1:,1], 0.0)
y_vals = 0.5*(y_vals[:-1] + y_vals[1:])
y_vals[segments_count-1] = y_vals[segments_count]

boundary_mesh_size = 2.0**((box_height - y_vals)*0.9e-2) + 0.4

domain_mesh_size = boundary_mesh_size.max()*1.1

fields = []

with pygmsh.geo.Geometry() as geom:
    poly = geom.add_polygon(polygon, mesh_size=domain_mesh_size)

    for i in range(2, segments_count*2-1):
        fields.append(geom.add_boundary_layer(edges_list=[poly.curves[i]],
                                              lcmin=boundary_mesh_size[i-1],
                                              lcmax=domain_mesh_size,
                                              distmin=0.0, distmax=5.0))

    geom.set_background_mesh(fields, operator='Min')
    mesh = geom.generate_mesh()

points = mesh.points[:,:2]
mesh_cells = np.array(mesh.cells_dict['triangle'], dtype='int64')

for i in range(1):
    points, mesh_cells = optimize_points_cells(points, mesh_cells, 'odt-cg',
                                                   tol=1e-5, max_num_steps=3,
                                                   verbose=True, omega=1.0)

    points, mesh_cells = optimize_points_cells(points, mesh_cells, "odt-fixed-point",
                                               tol=1e-5, max_num_steps=25,
                                               verbose=True, omega=1.0)

meshio_mesh = meshio.Mesh(points=points[:,:2], cells={'triangle': mesh_cells})
meshio_mesh.write(filepath)

fig, ax = plt.subplots(1, 1, figsize=(6, 10))
plt.triplot(points[:,0], points[:,1], mesh_cells)
ax.set_aspect('equal')
plt.show()
