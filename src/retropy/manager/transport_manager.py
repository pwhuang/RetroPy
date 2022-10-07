# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.problem import TracerTransportProblemExp
from retropy.physics import DG0Kernel

from dolfin import Constant, as_vector
from numpy import exp

class TransportManager(TracerTransportProblemExp, DG0Kernel):
    """Manages the default behavior of solving species transport."""

    def __init__(self, mesh, boundary_markers, domain_markers):
        super().__init__(mesh, boundary_markers, domain_markers)
        self.is_same_diffusivity = False

    def add_physics_to_form(self, u, theta_val=0.5, f_id=0):
        """
        Explicit upwind advection and Crank-Nicolson diffusion.
        """

        self.set_advection_velocity()

        theta = Constant(theta_val)
        one = Constant(1.0)

        self.add_explicit_advection(u, kappa=one, marker=0, f_id=f_id)

        for component in self.component_dict.keys():
            self.add_implicit_diffusion(component, kappa=theta, marker=0)
            self.add_explicit_diffusion(component, u, kappa=one-theta, marker=0)

        if self.is_same_diffusivity==False:
            self.add_implicit_charge_balanced_diffusion(kappa=theta, marker=0)
            self.add_explicit_charge_balanced_diffusion(u, kappa=one-theta, marker=0)

        self.evaluate_jacobian(self.get_forms()[0])

    def set_advection_velocity(self):
        super().set_advection_velocity()

    def solve_solvent_transport(self):
        fluid_comp_old = self.fluid_components.vector()
        fluid_comp_new = self.get_solution().vector()

        self.solvent.vector()[:] += \
        ((exp(fluid_comp_old) - exp(fluid_comp_new)).reshape(-1, self.num_component)\
        *self._M_fraction).sum(axis=1)

        # TODO: Test the effectiveness of the following script compared to the
        # implementation above.

        # self.solvent.vector()[:] += \
        # matmul((exp(fluid_comp_old) - exp(fluid_comp_new)).reshape(-1, self.num_component), \
        # self._M_fraction)

    def setup_transport_solver(self):
        self.generate_solver(eval_jacobian=False)
        self.set_solver_parameters('gmres', 'amg')

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        prm = self.get_solver_parameters()

        prm['nonlinear_solver'] = 'snes'

        nl_solver_type = 'snes_solver'

        prm[nl_solver_type]['absolute_tolerance'] = 1e-11
        prm[nl_solver_type]['relative_tolerance'] = 1e-13
        prm[nl_solver_type]['solution_tolerance'] = 1e-18
        prm[nl_solver_type]['maximum_iterations'] = 50
        prm['snes_solver']['method'] = 'newtonls'
        prm['snes_solver']['line_search'] = 'bt'
        prm['newton_solver']['relaxation_parameter'] = 0.1
        prm[nl_solver_type]['linear_solver'] = linear_solver
        prm[nl_solver_type]['preconditioner'] = preconditioner

        self.__set_krylov_solver_params(prm[nl_solver_type]['krylov_solver'])

    def __set_krylov_solver_params(self, prm):
        prm['absolute_tolerance'] = 1e-13
        prm['relative_tolerance'] = 1e-12
        prm['maximum_iterations'] = 2000
        prm['error_on_nonconvergence'] = False
        prm['monitor_convergence'] = False
        prm['nonzero_initial_guess'] = True
        prm['divergence_limit'] = 1e6
