# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedRectangleMesh
from dolfinx.fem import Function, Constant, assemble_scalar, form
from petsc4py.PETSc import ScalarType
from ufl import inner

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


class DarcyFlowBenchmark:
    """This benchmark problem is based on Ada Johanne Ellingsrud's masters
    thesis: Preconditioning unified mixed discretization of coupled Darcy-
    Stokes flow. https://www.duo.uio.no/bitstream/handle/10852/45338/paper.pdf
    """

    def get_mesh_and_markers(self, nx):
        marked_mesh = MarkedRectangleMesh()
        marked_mesh.set_bottom_left_coordinates(coord_x=-1.0, coord_y=-1.0)
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
        self.set_gravity((0.0, 0.0))

    def set_boundary_conditions(self):
        zero = Constant(self.mesh, ScalarType(0.0))
        self.set_pressure_bc({"left": zero, "right": zero})

        velocity_bc = Function(self.velocity_func_space)
        velocity_bc.interpolate(lambda x: (np.sin(np.pi * x[1]), np.cos(np.pi * x[0])))
        self.set_velocity_bc({"top": velocity_bc, "bottom": velocity_bc})

    def set_momentum_sources(self):
        momentum1 = Function(self.velocity_func_space)
        momentum1.interpolate(lambda x: (np.sin(np.pi * x[1]), np.cos(np.pi * x[0])))

        momentum2 = Function(self.velocity_func_space)
        momentum2.interpolate(
            lambda x: (
                np.pi * np.exp(x[1]) * np.cos(np.pi * x[0]),
                np.exp(x[1]) * np.sin(np.pi * x[0]),
            )
        )

        momentum_sources = [self._mu / self._k * momentum1 + momentum2]

        self.add_momentum_source(momentum_sources)

    def get_solution(self):
        self.sol_pressure = Function(self.pressure_func_space)
        self.sol_pressure.interpolate(lambda x: np.exp(x[1]) * np.sin(np.pi * x[0]))

        self.sol_velocity = Function(self.velocity_func_space)
        self.sol_velocity.interpolate(
            lambda x: (np.sin(np.pi * x[1]), np.cos(np.pi * x[0]))
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
