# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from abc import ABC, abstractmethod
from retropy.problem import DarcyFlowUzawa
from dolfinx.fem import Function

class DarcyFlowManagerUzawa(ABC, DarcyFlowUzawa):
    """Manager class that solves Darcy flow using Uzawa's method."""

    def setup_flow_solver(self, r_val=1e6, omega_by_r=1.0):
        self.set_flow_fe_space()
        self.set_fluid_properties()

        self.generate_form()
        self.generate_residual_form()
        self.set_flow_ibc()

        self.set_additional_parameters(r_val=r_val, omega_by_r=omega_by_r)
        self.assemble_matrix()
        self.set_flow_solver_params()

    def set_flow_fe_space(self):
        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('RT', 1)

    def set_flow_ibc(self):
        """Sets the initial and boundary conditions of the flow."""

        self.set_pressure_ic(lambda x: 0.0 * x[0])
        self.set_pressure_bc({})

        velocity_bc = Function(self.velocity_func_space)
        velocity_bc.x.array[:] = 0.0
        velocity_bc.x.scatter_forward()
        self.set_velocity_bc({'top': velocity_bc,
                              'right': velocity_bc,
                              'bottom': velocity_bc,
                              'left': velocity_bc})

    @abstractmethod
    def set_fluid_properties(self):
        pass

