# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedRectangleMesh
from retropy.problem import TransportProblemBase
from dolfinx.fem import Function, assemble_scalar, form, Constant

from ufl import inner

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


class DarcyBoundarySource(TransportProblemBase):
    """This benchmark problem tests a Darcy flow problem specifying flux boundary
    at the lnlet and pressure boundary at the outlet.
    """

    def get_mesh_and_markers(self, nx):
        marked_mesh = MarkedRectangleMesh()
        marked_mesh.set_bottom_left_coordinates(coord_x=0.0, coord_y=0.0)
        marked_mesh.set_top_right_coordinates(coord_x=1.0, coord_y=1.0)
        marked_mesh.set_number_of_elements(nx, nx)
        marked_mesh.set_mesh_type("triangle")
        marked_mesh.locate_and_mark_boundaries()

        marked_mesh.generate_mesh(mesh_shape="left/right")
        marked_mesh.generate_boundary_markers()
        marked_mesh.generate_interior_markers()
        marked_mesh.generate_domain_markers()

        self.mesh_characteristic_length = 1.0 / nx

        return marked_mesh

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_material_properties(self):
        self.set_porosity(1.0)
        self.set_permeability(1.0)
        self.set_fluid_density(1.0)
        self.set_fluid_viscosity(1.0)
        self.set_gravity((0.0, -1.0))

    def set_boundary_conditions(self, penalty_value):
        pressure_bc = self.fluid_density * self._g[1] * self.cell_coord[1]
        self.set_pressure_bc({"right": pressure_bc})
        self.add_weak_pressure_bc(penalty_value)

        velocity_bc = Function(self.velocity_func_space)
        velocity_bc.interpolate(lambda x: (1.0 + 0.0*x[0], 0.0*x[1]))

        zero_bc = Function(self.velocity_func_space)
        zero_bc.interpolate(lambda x: (0.0*x[0], 0.0*x[1]))

        self.set_velocity_bc({"top": zero_bc, "bottom": zero_bc,
                              "left": velocity_bc})

    def set_momentum_sources(self):
        pass

    def get_solution(self):
        self.sol_pressure = Function(self.pressure_func_space)
        self.sol_pressure.interpolate(lambda x: - x[1] - x[0] + 1.0)

        self.sol_velocity = Function(self.velocity_func_space)
        self.sol_velocity.interpolate(
            lambda x: (1.0 + 0.0 * x[0], 0.0 * x[1])
        )

        return self.sol_pressure, self.sol_velocity

    def get_error_norm(self):
        p_error = self.fluid_pressure - self.sol_pressure
        v_error = self.fluid_velocity - self.sol_velocity

        p_error_local = assemble_scalar(form(p_error**2 * self.dx))
        v_error_local = assemble_scalar(form(inner(v_error, v_error) * self.dx))

        comm = self.mesh.comm
        pressure_error_global = np.sqrt(comm.allreduce(p_error_local, op=MPI.SUM))
        velocity_error_global = np.sqrt(comm.allreduce(v_error_local, op=MPI.SUM))

        return pressure_error_global, velocity_error_global

    def mpl_output(self):
        x_DG, y_DG, _ = self.DG0_space.tabulate_dof_coordinates().T
        x, y, _ = self.Vec_CG1_space.tabulate_dof_coordinates().T

        v, num_v = Function(self.Vec_CG1_space), Function(self.Vec_CG1_space)
        v.interpolate(self.sol_velocity)
        num_v.interpolate(self.fluid_velocity)

        vx, vy = v.x.array.reshape(-1, 2).T
        num_vx, num_vy = num_v.x.array.reshape(-1, 2).T

        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].tricontourf(x_DG, y_DG, self.sol_pressure.x.array)
        ax[0].quiver(x, y, vx, vy)
        ax[1].tricontourf(x_DG, y_DG, self.fluid_pressure.x.array)
        ax[1].quiver(x, y, num_vx, num_vy)
        plt.show()
