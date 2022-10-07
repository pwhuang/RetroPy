# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from abc import ABC, abstractmethod
from retropy.problem import DarcyFlowUzawa
from dolfin import PETScLUSolver, PETScKrylovSolver, Constant

class DarcyFlowManagerUzawa(ABC, DarcyFlowUzawa):
    """Manager class that solves Darcy flow using Uzawa's method."""

    def setup_flow_solver(self, r_val=1e6, omega_by_r=1.0):
        self.set_flow_fe_space()
        self.set_flow_ibc()

        self.set_fluid_properties()

        self.generate_form()
        self.generate_residual_form()

        self.set_flow_solver_params()
        self.set_additional_parameters(r_val=r_val, omega_by_r=omega_by_r)
        self.assemble_matrix()

    def set_flow_fe_space(self):
        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('RT', 1)

    @abstractmethod
    def set_flow_ibc(self):
        """Sets the initial and boundary conditions of the flow."""

        self.mark_flow_boundary(pressure = [],
                                velocity = [1, 2, 3, 4])

        self.set_pressure_bc([])
        self.set_pressure_ic(Constant(0.0))
        self.set_velocity_bc([Constant([0.0, 0.0])]*4)

    @abstractmethod
    def set_fluid_properties(self):
        pass

    def set_flow_solver_params(self):
        self.solver_v = PETScLUSolver('mumps')
        self.solver_p = PETScKrylovSolver('gmres', 'none')

        prm_v = self.solver_v.parameters
        prm_p = self.solver_p.parameters

        # prm_v['symmetric'] = True
        self.__set_krylov_solver_params(prm_p)

    def __set_krylov_solver_params(self, prm):
        prm['absolute_tolerance'] = 1e-10
        prm['relative_tolerance'] = 1e-14
        prm['maximum_iterations'] = 8000
        prm['error_on_nonconvergence'] = True
        prm['monitor_convergence'] = False
        prm['nonzero_initial_guess'] = True
