# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *


class TransportProblemBase:
    """Base class for all problems that use FeNiCs."""

    def set_mesh(self, mesh):
        """Setup mesh and define mesh related quantities."""

        self.mesh = mesh

        self.n = FacetNormal(self.mesh)
        self.facet_area = FacetArea(self.mesh)
        self.cell_volume = CellVolume(self.mesh)

        self.set_periodic_bcs(bcs=None)
        self.__init_TPFA_recipe(option="cell_centered")

    def __init_TPFA_recipe(self, option):
        # TODO: Find a better name for this method.
        mesh_dim = self.mesh.topology.dim
        facet_dim = mesh_dim - 1
        cell_vertex_num = self.mesh.ufl_cell().num_vertices()

        self.DG0_space = functionspace(self.mesh, ("DG", 0))
        self.CG1_space = functionspace(self.mesh, ("CG", 1))

        self.Vec_DG0_space = functionspace(self.mesh, ("DG", 0, (mesh_dim,)))
        self.Vec_CG1_space = functionspace(self.mesh, ("CG", 1, (mesh_dim,)))

        # The implementation of boundary_vertex_coord potentially leads to a
        # erroneous boundary diffusion flux approximation for triangular meshes.
        # It is wrong on the corners of quad meshes.
        # TODO: Please address this.

        self.mesh.topology.create_connectivity(facet_dim, mesh_dim)
        boundary_facets = exterior_facet_indices(self.mesh.topology)
        boundary_dofs = locate_dofs_topological(
            self.Vec_CG1_space, facet_dim, boundary_facets
        )

        self.vertex_coord = Function(self.Vec_CG1_space)
        self.vertex_coord.interpolate(lambda x: [x[i] for i in range(mesh_dim)])

        self.cell_coord = Function(self.Vec_DG0_space)

        if option == "cell_centered":
            self.cell_coord.interpolate(self.vertex_coord)

        self.delta_h = sqrt(dot(jump(self.cell_coord), jump(self.cell_coord)))

        self.boundary_vertex_coord = Function(self.Vec_CG1_space)
        petsc.set_bc(
            self.boundary_vertex_coord.x.petsc_vec,
            bcs=[dirichletbc(self.vertex_coord, boundary_dofs)],
        )
        self.boundary_vertex_coord.x.scatter_forward()

        self.boundary_cell_coord = Function(self.Vec_DG0_space)
        self.boundary_cell_coord.interpolate(self.boundary_vertex_coord)
        self.boundary_cell_coord.x.petsc_vec.array_w *= cell_vertex_num / mesh_dim

    def set_boundary_markers(self, boundary_markers):
        self.boundary_markers = boundary_markers

        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundary_markers)

    def set_interior_markers(self, interior_markers):
        self.interior_markers = interior_markers

        self.dS = Measure("dS", domain=self.mesh, subdomain_data=self.interior_markers)

    def set_domain_markers(self, domain_markers):
        self.domain_markers = domain_markers

        self.dx = Measure("dx", domain=self.mesh, subdomain_data=self.domain_markers)

    def set_periodic_bcs(self, bcs=None):
        self.periodic_bcs = bcs

    def set_velocity_vector_fe_space(self, fe_space, fe_degree):
        cellname = self.mesh.ufl_cell().cellname()
        mesh_dim = self.mesh.geometry.dim
        self.velocity_finite_element = element(
            fe_space, cellname, fe_degree, shape=(mesh_dim,), dtype=default_real_type
        )

        self.velocity_func_space = functionspace(
            self.mesh, self.velocity_finite_element
        )

        self.fluid_velocity = Function(self.velocity_func_space)
        self.fluid_velocity.name = "velocity"

    def set_velocity_fe_space(self, fe_space, fe_degree):
        cellname = self.mesh.ufl_cell().cellname()
        self.velocity_finite_element = element(
            fe_space, cellname, fe_degree, dtype=default_real_type
        )

        self.velocity_func_space = functionspace(
            self.mesh, self.velocity_finite_element
        )

        self.fluid_velocity = Function(self.velocity_func_space)
        self.fluid_velocity.name = "velocity"

    def set_pressure_fe_space(self, fe_space, fe_degree):
        cellname = self.mesh.ufl_cell().cellname()
        self.pressure_finite_element = element(
            fe_space, cellname, fe_degree, dtype=default_real_type
        )

        self.pressure_func_space = functionspace(
            self.mesh, self.pressure_finite_element
        )

        self.fluid_pressure = Function(self.pressure_func_space)
        self.fluid_pressure.name = "pressure"

    def get_fluid_velocity(self):
        return self.fluid_velocity

    def get_fluid_pressure(self):
        return self.fluid_pressure

    def quick_save(self, file_name):
        """"""
        with XDMFFile(self.mesh.comm, file_name + ".xdmf", "w") as obj:
            obj.write_mesh(self.mesh)
            for func in self.functions_to_save:
                obj.write_function(func, t=0)

    def save_fluid_pressure(self, time_step):
        self.write_function(self.fluid_pressure, time_step)

    def save_fluid_velocity(self, time_step):
        self.write_function(self.fluid_velocity, time_step)

    @staticmethod
    def set_default_solver_parameters(prm):
        prm["absolute_tolerance"] = 1e-14
        prm["relative_tolerance"] = 1e-12
        prm["maximum_iterations"] = 5000
        prm["error_on_nonconvergence"] = True
        prm["monitor_convergence"] = True
        prm["nonzero_initial_guess"] = True

    @staticmethod
    def circumcenter_from_points(ax, ay, bx, by, cx, cy):
        # Translation considering A as the origin of the Cartesian coordinate
        bxt = bx - ax
        byt = by - ay

        cxt = cx - ax
        cyt = cy - ay

        D = 2.0 * (bxt * cyt - byt * cxt)
        bt_l2 = bxt**2 + byt**2  # length squared
        ct_l2 = cxt**2 + cyt**2

        ux = (cyt * bt_l2 - byt * ct_l2) / D + ax
        uy = (bxt * ct_l2 - cxt * bt_l2) / D + ay

        return [ux, uy]
