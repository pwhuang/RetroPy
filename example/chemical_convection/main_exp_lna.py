import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '../../')

from mesh_factory import MeshFactory
from flow_manager_uzawa import FlowManager
from transport_manager_exp import TransportManager
from reaction_manager import ReactionManager
from aux_variables import AuxVariables
from reaktoro_transport.tests import quick_plot

import numpy as np
from dolfin import info, begin, end, as_vector, Constant

class main(FlowManager, TransportManager, ReactionManager, MeshFactory, AuxVariables):
    """
    This class solves the chemically driven convection problem.
    """

    def __init__(self, nx, ny):
        TransportManager.__init__(self, *self.get_mesh_and_markers(nx, ny))

    def solve_initial_condition(self):
        self.assign_u0_to_u1()
        self._solve_chem_equi_over_dofs()
        #self._rho_old.assign(self.fluid_density)

    def set_advection_velocity(self):
        D = self.molecular_diffusivity
        advection_list = []

        for i in range(self.num_component):
            advection_list.append(self.fluid_velocity - Constant(D[i])*self._grad_lna.sub(i))

        self.advection_velocity = as_vector(advection_list)

    def _save_fluid_density(self, time):
        self.xdmf_obj.write_checkpoint(self.fluid_density,
                                       self.fluid_density.name(),
                                       time_step=time,
                                       append=True)

    def _save_log_activity_coeff(self, time):
        self._save_mixed_function(time, self.ln_activity, self.ln_activity_dict)

    def save_to_file(self, time):
        super().save_to_file(time, is_saving_pv=True)
        self._save_fluid_density(time)
        self._save_log_activity_coeff(time)
        self._save_auxiliary_variables(time)

    def timestepper(self):
        pass

    def solve(self, dt_val=1.0, endtime=10.0):
        self.solve_initial_condition()
        self.solve_flow(target_residual=1e-10, max_steps=30)
        self.solve_projection()
        self.solve_auxiliary_variables()

        r_val = 3e6
        current_time = 0.0
        min_dt = 1e-7
        max_dt = 2.0
        timestep = 1
        self.save_to_file(time=current_time)

        while current_time < endtime:
            info('timestep = ' + str(timestep) + ',  dt = ' + str(dt_val)\
                 + ', current_time = ' + str(current_time) )

            self.set_dt(dt_val)

            try:
                self.solve_one_step()
            except:
                self.assign_u0_to_u1()
                if (dt_val := 0.7*dt_val) < min_dt:
                    print('Reached minimum dt. Abort!')
                    break
                continue

            self.solve_solvent_amount(self.get_solution())

            self._solve_chem_equi_over_dofs()
            self.assign_u1_to_u0()

            self.solve_flow(target_residual=1e-10, max_steps=10)
            self.solve_projection()
            self.solve_auxiliary_variables()

            timestep += 1
            current_time += dt_val

            if (dt_val := dt_val*1.1) > max_dt:
                dt_val = max_dt

            if timestep%1 == 0:
                self.save_to_file(time=current_time)


problem = main(nx=24, ny=40)
problem.generate_output_instance(sys.argv[1])
problem.define_problem()

problem.setup_flow_solver()
problem.setup_reaction_solver()
problem.setup_projection_solver()
problem.setup_transport_solver()

problem.setup_auxiliary_solver()

problem.solve(dt_val=1e-1, endtime=500.0)
