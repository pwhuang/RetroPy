from . import *

class TransportProblemBase():
    """Base class for all problems that use FeNiCs."""

    def __init__(self):
        pass

    def set_mesh(self, mesh):
        self.mesh = mesh

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
                                                     self.mesh.ufl_cell(),
                                                     fe_degree)

    def set_pressure_fe_space(self, fe_space: str, fe_degree: int):
        self.pressure_finite_element = FiniteElement(fe_space,
                                                     self.mesh.ufl_cell(),
                                                     fe_degree)

    @staticmethod
    def set_default_solver_parameters(prm):
        prm['absolute_tolerance'] = 1e-14
        prm['relative_tolerance'] = 1e-12
        prm['maximum_iterations'] = 2000
        prm['error_on_nonconvergence'] = True
        prm['monitor_convergence'] = True
        prm['nonzero_initial_guess'] = True
