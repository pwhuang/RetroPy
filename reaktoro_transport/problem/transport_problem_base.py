from . import *

class TransportProblemBase():
    """Base class for all problems that use FeNiCs."""

    def __init__(self):
        pass

    def set_mesh(self, mesh):
        """Setup mesh and define mesh related quantities."""

        self.mesh = mesh

        self.n = FacetNormal(self.mesh)

        DG0_space = FunctionSpace(mesh, 'DG', 0)

        self.space_dim = []
        for i in range(self.mesh.geometric_dimension()):
            self.space_dim.append(
            interpolate(Expression('x[' + str(i) + ']', degree=0), DG0_space))

        self.delta_h = sqrt(sum([jump(x)**2 for x in self.space_dim]))

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

    def set_velocity_fe_space(self, fe_space: str, fe_degree: int):
        self.velocity_finite_element = VectorElement(fe_space,
                                                     self.mesh.cell_name(),
                                                     fe_degree)

        self.velocity_func_space = FunctionSpace(self.mesh,
                                                 self.velocity_finite_element)

        self.fluid_velocity = Function(self.velocity_func_space)

    def set_pressure_fe_space(self, fe_space: str, fe_degree: int):
        self.pressure_finite_element = FiniteElement(fe_space,
                                                     self.mesh.cell_name(),
                                                     fe_degree)

        self.pressure_func_space = FunctionSpace(self.mesh,
                                                 self.pressure_finite_element)

        self.fluid_pressure = Function(self.pressure_func_space)

    def get_fluid_velocity(self):
        return self.fluid_velocity.copy()

    def get_fluid_pressure(self):
        return self.fluid_pressure.copy()

    def quick_save(self, file_name: str):
        """"""
        with XDMFFile(file_name + '.xdmf') as obj:
            obj.parameters['flush_output'] = True
            obj.write(self.mesh)
            for func in self.functions_to_save:
                obj.write_checkpoint(func, func.name(),
                                     time_step=0, append=True)

    def set_output_instance(self, file_name: str):
        self.xdmf_obj = XDMFFile(file_name + '.xdmf')
        self.xdmf_obj.write(self.mesh)

    def delete_output_instance(self):
        self.xdmf_obj.close()

    @staticmethod
    def set_default_solver_parameters(prm):
        prm['absolute_tolerance'] = 1e-14
        prm['relative_tolerance'] = 1e-12
        prm['maximum_iterations'] = 2000
        prm['error_on_nonconvergence'] = True
        prm['monitor_convergence'] = True
        prm['nonzero_initial_guess'] = True
