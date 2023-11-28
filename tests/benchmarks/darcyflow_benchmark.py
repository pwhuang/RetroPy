# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedRectangleMesh
from dolfinx.fem import Function, assemble_scalar, form, FunctionSpace
from ufl import inner

from mpi4py import MPI
import numpy as np

class DarcyFlowBenchmark:
    """This benchmark problem is based on Ada Johanne Ellingsrud's masters
    thesis: Preconditioning unified mixed discretization of coupled Darcy-
    Stokes flow. https://www.duo.uio.no/bitstream/handle/10852/45338/paper.pdf
    """

    def get_mesh_and_markers(self, nx):
        marked_mesh = MarkedRectangleMesh()
        marked_mesh.set_bottom_left_coordinates(coord_x = -1.0, coord_y = -1.0)
        marked_mesh.set_top_right_coordinates(coord_x = 1.0, coord_y = 1.0)
        marked_mesh.set_number_of_elements(nx, nx)
        marked_mesh.set_mesh_type('triangle')
        marked_mesh.locate_and_mark_boundaries()

        marked_mesh.generate_mesh(mesh_shape='left/right')
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
        self.mark_flow_boundary(pressure = [],
                                velocity = ['top', 'bottom'])
        
        self.generate_form()
        self.generate_residual_form()

        velocity_bc = Function(self.velocity_func_space)
        velocity_bc.interpolate(lambda x: (np.sin(np.pi * x[1]), np.cos(np.pi * x[0])))
        self.set_velocity_bc([velocity_bc]*2)

    def set_momentum_sources(self):
        momentum1 = Function(self.velocity_func_space)
        momentum1.interpolate(lambda x: (np.sin(np.pi * x[1]), np.cos(np.pi * x[0])))

        momentum2 = Function(self.velocity_func_space)
        momentum2.interpolate(lambda x: (np.pi * np.exp(x[1]) * np.cos(np.pi * x[0]),
                                         np.exp(x[1]) * np.sin(np.pi * x[0])))

        momentum_sources = [self._mu / self._k * momentum1 + momentum2]
        
        self.add_momentum_source(momentum_sources)
        self.add_momentum_source_to_residual_form(momentum_sources)

    def get_solution(self):
        self.sol_pressure = Function(self.pressure_func_space)
        self.sol_pressure.interpolate(lambda x: np.exp(x[1]) * np.sin(np.pi*x[0]))

        self.sol_velocity = Function(self.velocity_func_space)
        self.sol_velocity.interpolate(lambda x: (np.sin(np.pi*x[1]), np.cos(np.pi*x[0])))

        return self.sol_pressure, self.sol_velocity

    def get_error_norm(self):
        pressure_error = self.fluid_pressure - self.sol_pressure
        velocity_error = self.fluid_velocity - self.sol_velocity

        pressure_error_local = assemble_scalar(form(pressure_error ** 2 * self.dx))
        velocity_error_local = assemble_scalar(form(inner(velocity_error, velocity_error) * self.dx))

        comm = self.mesh.comm
        pressure_error_global = np.sqrt(comm.allreduce(pressure_error_local, op=MPI.SUM))
        velocity_error_global = np.sqrt(comm.allreduce(velocity_error_local, op=MPI.SUM))

        return pressure_error_global, velocity_error_global
