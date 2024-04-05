# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from mesh_factory import MeshFactory
from retropy.manager import DarcyFlowManagerUzawa as FlowManager
from retropy.manager import ReactiveTransportManager
from retropy.manager import XDMFManager as OutputManager
from retropy.solver import TransientNLSolver

from dolfinx.fem import Expression
from ufl import SpatialCoordinate, conditional, lt

class FlowManager(FlowManager):
    def set_flow_fe_space(self):
        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('RTCF', 1)

class Problem(ReactiveTransportManager, FlowManager, OutputManager,
              TransientNLSolver):
    """This class solves the chemically driven convection problem."""

    def __init__(self, nx, ny, const_diff):
        super().__init__(MeshFactory(nx, ny))
        self.is_same_diffusivity = const_diff
        self.set_flow_residual(5e-10)

    def set_component_properties(self):
        self.set_molar_mass([22.98977, 62.0049, 1.00794, 17.00734]) #g/mol
        self.set_solvent_molar_mass(18.0153)
        self.set_charge([1.0, -1.0, 1.0, -1.0])

    def define_problem(self):
        self.set_components('Na+ NO3- H+ OH-')
        self.set_solvent('H2O(aq)')
        self.set_component_properties()

        self.set_component_fe_space()
        self.initialize_form()

        self.background_pressure = 1e5 + 1e-3*9806.65*45 # Pa

        HNO3_amounts = [1e-15, 1.5, 1.5, 1e-13, 52.712] # micro mol/mm^3
        NaOH_amounts = [1.4, 1e-15, 1e-15, 1.4, 55.361]

        x = SpatialCoordinate(self.mesh)

        for i, comp in enumerate(self.component_dict.keys()):
            ic_expr = Expression(conditional(lt(x[1], 45.0), NaOH_amounts[i], HNO3_amounts[i]),
                                 self.comp_func_spaces.sub(i).element.interpolation_points())
            self.set_component_ics(comp, ic_expr)

        ic_expr = Expression(conditional(lt(x[1], 45.0), NaOH_amounts[-1], HNO3_amounts[-1]),
                             self.DG0_space.element.interpolation_points())
        self.set_solvent_ic(ic_expr)

    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity((1.232e-3 + 0.933e-3)*0.5)  # Pa sec
        self.set_gravity([0.0, -9806.65]) # mm/sec
        self.set_permeability(1.2**2/12.0) # mm^2

    @staticmethod
    def timestepper(dt_val, current_time, time_stamp):
        min_dt, max_dt = 1e-3, 1.0

        if (dt_val := dt_val*1.1) > max_dt:
            dt_val = max_dt
        elif dt_val < min_dt:
            dt_val = min_dt
        if dt_val > time_stamp - current_time:
            dt_val = time_stamp - current_time

        return dt_val
