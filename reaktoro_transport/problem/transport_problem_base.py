from . import *
from ..mesh import MarkerCollection

class TransportProblemBase():
    """Base class for all problems that use FeNiCs."""

    def set_mesh(self, mesh):
        """Setup mesh and define mesh related quantities."""

        self.mesh = mesh

        self.n = FacetNormal(self.mesh)
        self.facet_area = FacetArea(self.mesh)
        self.cell_volume = CellVolume(self.mesh)

        self.DG0_space = FunctionSpace(self.mesh, 'DG', 0)
        self.Vec_DG0_space = VectorFunctionSpace(self.mesh, 'DG', 0)
        self.Vec_CG1_space = VectorFunctionSpace(self.mesh, 'CG', 1)

        space_list = []

        for i in range(self.mesh.geometric_dimension()):
            space_list.append('x[' + str(i) + ']')

        # Consider move this part of code to other class method?
        # TODO: Yeah, we should move it!
        # The implementation of boundary_vertex_coord potentially leads to a
        # erroneous boundary diffusion flux approximation (only for triangles).
        # TODO: Please address this.

        self.vertex_coord = interpolate(Expression(space_list, degree=1), self.Vec_CG1_space)
        self.cell_coord = interpolate(Expression(space_list, degree=0), self.Vec_DG0_space)

        self.delta_h = sqrt(dot(jump(self.cell_coord), jump(self.cell_coord)))

        bc = DirichletBC(self.Vec_CG1_space, [1]*self.mesh.geometric_dimension(),
                         MarkerCollection.AllBoundary())
        boundary_mask = Function(self.Vec_CG1_space)
        self.boundary_vertex_coord = Function(self.Vec_CG1_space)

        bc.apply(boundary_mask.vector())
        self.boundary_vertex_coord.vector()[:] = boundary_mask.vector()[:] \
                                                 *self.vertex_coord.vector()[:]
        self.boundary_cell_coord = project(self.boundary_vertex_coord,
                                           self.Vec_DG0_space, solver_type='mumps')

    def set_boundary_markers(self, boundary_markers):
        self.boundary_markers = boundary_markers

        self.ds = Measure('ds', domain=self.mesh,
                          subdomain_data=self.boundary_markers)
        self.dS = Measure('dS', domain=self.mesh,
                          subdomain_data=self.boundary_markers)

    def set_domain_markers(self, domain_markers):
        self.domain_markers = domain_markers

        self.dx = Measure('dx', domain=self.mesh,
                          subdomain_data=self.domain_markers)

    def set_velocity_vector_fe_space(self, fe_space: str, fe_degree: int):
        self.velocity_finite_element = VectorElement(fe_space,
                                                     self.mesh.cell_name(),
                                                     fe_degree)

        self.velocity_func_space = FunctionSpace(self.mesh,
                                                 self.velocity_finite_element)

        self.fluid_velocity = Function(self.velocity_func_space)
        self.fluid_velocity.rename('velocity', 'fluid velocity')

    def set_velocity_fe_space(self, fe_space: str, fe_degree: int):
        self.velocity_finite_element = FiniteElement(fe_space,
                                                     self.mesh.cell_name(),
                                                     fe_degree)

        self.velocity_func_space = FunctionSpace(self.mesh,
                                                 self.velocity_finite_element)

        self.fluid_velocity = Function(self.velocity_func_space)
        self.fluid_velocity.rename('velocity', 'fluid velocity')

    def set_pressure_fe_space(self, fe_space: str, fe_degree: int):
        self.pressure_finite_element = FiniteElement(fe_space,
                                                     self.mesh.cell_name(),
                                                     fe_degree)

        self.pressure_func_space = FunctionSpace(self.mesh,
                                                 self.pressure_finite_element)

        self.fluid_pressure = Function(self.pressure_func_space)
        self.fluid_pressure.rename('pressure', 'fluid pressure')

    def get_fluid_velocity(self):
        return self.fluid_velocity

    def get_fluid_pressure(self):
        return self.fluid_pressure

    def quick_save(self, file_name: str):
        """"""
        with XDMFFile(file_name + '.xdmf') as obj:
            obj.parameters['flush_output'] = True
            obj.write(self.mesh)
            for func in self.functions_to_save:
                obj.write_checkpoint(func, func.name(),
                                     time_step=0, append=True)

    def save_fluid_pressure(self, time_step, is_appending):
        self.xdmf_obj.write_checkpoint(self.fluid_pressure,
                                       self.fluid_pressure.name(),
                                       time_step=time_step,
                                       append=is_appending)

    def save_fluid_velocity(self, time_step, is_appending):
        self.xdmf_obj.write_checkpoint(self.fluid_velocity,
                                       self.fluid_velocity.name(),
                                       time_step=time_step,
                                       append=is_appending)

    def generate_output_instance(self, file_name: str):
        self.xdmf_obj = XDMFFile(MPI.comm_world, file_name + '.xdmf')
        self.xdmf_obj.write(self.mesh)

        self.xdmf_obj.parameters['flush_output'] = True
        self.xdmf_obj.parameters['functions_share_mesh'] = True
        self.xdmf_obj.parameters['rewrite_function_mesh'] = False

        return True

    def delete_output_instance(self):
        try:
            self.xdmf_obj
        except:
            return False

        self.xdmf_obj.close()
        return True

    @staticmethod
    def set_default_solver_parameters(prm):
        prm['absolute_tolerance'] = 1e-14
        prm['relative_tolerance'] = 1e-12
        prm['maximum_iterations'] = 5000
        prm['error_on_nonconvergence'] = True
        prm['monitor_convergence'] = True
        prm['nonzero_initial_guess'] = True
