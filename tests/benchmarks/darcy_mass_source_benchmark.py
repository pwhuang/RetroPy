# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedRectangleMesh
from dolfinx.fem import Function, assemble_scalar, form
from ufl import inner

from mpi4py import MPI
import numpy as np

class DarcyMassSourceBenchmark:
    """This benchmark problem studies the effect of mass source (e.g., density
    change over time) on Darcy flow.
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

        self.mesh_characteristic_length = 1.0/nx

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
        self.set_pressure_bc({})
        self.generate_form()
        self.generate_residual_form()

        velocity_bc = Function(self.velocity_func_space)
        velocity_bc.x.array[:] = 0.0
        velocity_bc.x.scatter_forward()

        self.set_velocity_bc({'left': velocity_bc, 'right': velocity_bc, 'top': velocity_bc, 'bottom': velocity_bc})

    def set_mass_sources(self):
        mass_source = Function(self.pressure_func_space)
        mass_source.interpolate(lambda x: 2.0 * np.pi**2 * np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))
        # mass_sources = [Expression(('2*M_PI*M_PI*cos(M_PI*x[0])*cos(M_PI*x[1])'), degree=1)]
        mass_sources = [mass_source]

        self.add_mass_source(mass_sources)
        self.add_mass_source_to_residual_form(mass_sources)

    def get_solution(self):
        self.sol_pressure = Function(self.pressure_func_space)
        self.sol_pressure.interpolate(lambda x: np.cos(np.pi*x[0]) * np.cos(np.pi*x[1]))
        # self.sol_pressure = interpolate(Expression('cos(M_PI*x[0])*cos(M_PI*x[1])', degree=1),
        #                                 self.pressure_func_space)

        self.sol_velocity = Function(self.velocity_func_space)
        self.sol_velocity.interpolate(lambda x: (np.pi * np.sin(np.pi*x[0])*np.cos(np.pi*x[1]),
                                                 np.pi * np.cos(np.pi*x[0])*np.sin(np.pi*x[1])))

        # self.sol_velocity = interpolate(Expression(['M_PI*sin(M_PI*x[0])*cos(M_PI*x[1])',
        #                                             'M_PI*cos(M_PI*x[0])*sin(M_PI*x[1])'], degree=1),
        #                                 self.velocity_func_space)

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

