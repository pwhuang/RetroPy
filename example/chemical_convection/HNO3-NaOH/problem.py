# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from mesh_factory import MeshFactory
from retropy.manager import DarcyFlowManagerUzawa as FlowManager
from retropy.manager import ReactiveTransportManager
from retropy.manager import XDMFManager as OutputManager
from retropy.solver import TransientNLSolver

from dolfin import Expression, Constant, PETScOptions

class Problem(ReactiveTransportManager, FlowManager, MeshFactory, OutputManager,
              TransientNLSolver):
    """This class solves the chemically driven convection problem."""

    def __init__(self, nx, ny, const_diff):
        super().__init__(*self.get_mesh_and_markers(nx, ny))
        self.is_same_diffusivity = const_diff
        self.set_flow_residual(5e-10)

    def set_component_properties(self):
        self.set_molar_mass([22.98977, 62.0049, 1.00794, 17.00734]) #g/mol
        self.set_solvent_molar_mass(18.0153)
        self.set_charge([1.0, -1.0, 1.0, -1.0])

    def define_problem(self):
        self.set_components('Na+', 'NO3-', 'H+', 'OH-')
        self.set_solvent('H2O(l)')
        self.set_component_properties()

        self.set_component_fe_space()
        self.initialize_form()

        self.background_pressure = 1e5 + 1e-3*9806.65*45 # Pa

        HNO3_amounts = [1e-15, 1.5, 1.5, 1e-13, 52.712] # micro mol/mm^3
        NaOH_amounts = [1.4, 1e-15, 1e-15, 1.4, 55.361]

        init_expr_list = []

        for i in range(self.num_component):
            init_expr_list.append('x[1]<=45.0 ?' + str(NaOH_amounts[i]) + ':' + str(HNO3_amounts[i]))

        self.set_component_ics(Expression(init_expr_list, degree=1))
        self.set_solvent_ic(Expression('x[1]<=45.0 ?' + str(NaOH_amounts[-1]) + ':' + str(HNO3_amounts[-1]) , degree=1))

    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity((1.232e-3 + 0.933e-3)*0.5)  # Pa sec
        self.set_gravity([0.0, -9806.65]) # mm/sec
        self.set_permeability(1.2**2/12.0) # mm^2

    def set_flow_ibc(self):
        self.mark_flow_boundary(pressure = [],
                                velocity = [self.marker_dict['top'], self.marker_dict['bottom'],
                                            self.marker_dict['left'], self.marker_dict['right']])

        self.set_pressure_bc([]) # Pa
        self.set_pressure_ic(Constant(0.0))
        self.set_velocity_bc([Constant([0.0, 0.0])]*4)

    def setup_transport_solver(self):
        self.generate_solver(eval_jacobian=False)
        self.set_solver_parameters('bicgstab', 'amg')
        PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.4)
        PETScOptions.set("pc_hypre_boomeramg_truncfactor", 0.0)
        #PETScOptions.set("pc_hypre_boomeramg_print_statistics", 1)

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
