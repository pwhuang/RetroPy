import os
os.environ['OMP_NUM_THREADS'] = '1'

from mesh_factory import MeshFactory
from chemical_convection.flow_manager_uzawa import FlowManager
from chemical_convection.transport_manager import TransportManager
from chemical_convection.reaction_manager import ReactionManager
from chemical_convection.aux_variables import AuxVariables

import numpy as np
from dolfin import (info, DOLFIN_EPS, assemble, exp, begin, end, as_vector,
                    Expression, MPI)

class Problem(FlowManager, TransportManager, ReactionManager,
              MeshFactory, AuxVariables):
    """This class solves the chemically driven convection problem."""

    def __init__(self, nx, ny):
        TransportManager.__init__(self, *self.get_mesh_and_markers(nx, ny))
        self.__MPI_rank = MPI.rank(MPI.comm_world)

    def set_component_properties(self):
        self.set_molar_mass([22.99, 35.453, 1.0, 17.0]) #g/mol
        self.set_solvent_molar_mass(18.0)
        self.set_charge([1.0, -1.0, 1.0, -1.0])

    def define_problem(self):
        self.set_components('Na+', 'Cl-', 'H+', 'OH-')
        self.set_solvent('H2O(l)')
        self.set_component_properties()

        self.set_component_fe_space()
        self.initialize_form()

        self.background_pressure = 101325 + 1e-3*9806.65*25 # Pa

        HCl_amounts = [1e-13, 1.0, 1.0, 1e-13, 54.17] # micro mol/mm^3 # mol/L
        NaOH_amounts = [1.0, 1e-13, 1e-13, 1.0, 55.36]

        init_expr_list = []

        for i in range(self.num_component):
            init_expr_list.append('x[1]<=25.0 ?' + str(NaOH_amounts[i]) + ':' + str(HCl_amounts[i]))

        self.set_component_ics(Expression(init_expr_list, degree=1))
        self.set_solvent_ic(Expression('x[1]<=25.0 ?' + str(NaOH_amounts[-1]) + ':' + str(HCl_amounts[-1]) , degree=1))

    def set_fluid_properties(self):
        super().set_fluid_properties()
        self.set_permeability(0.5**2/12.0) # mm^2

    def solve_initial_condition(self):
        self.assign_u0_to_u1()
        self._solve_chem_equi_over_dofs()

    def set_advection_velocity(self):
        self.advection_velocity = \
        as_vector([self.fluid_velocity for i in range(self.num_component)])

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

    def solve(self, dt_val=1.0, endtime=10.0, time_stamps=[]):
        self.solve_initial_condition()
        self.solve_flow(target_residual=1e-10, max_steps=30)
        #self.solve_projection()
        self.solve_auxiliary_variables()

        r_val = 3e6
        current_time = 0.0
        min_dt = 5e-2
        max_dt = 2.0
        timestep = 1
        saved_times = []

        max_trials = 7
        trial_count = 0

        time_stamps.append(endtime)
        time_stamp = time_stamps.pop(0)

        self.save_to_file(time=current_time)
        saved_times.append(current_time)

        while current_time < endtime:
            if self.__MPI_rank==0:
                info(f"timestep = {timestep}, dt = {dt_val:.6f}, "\
                     f"current_time = {current_time:.6f}\n")

            self.set_dt(dt_val)

            try:
                self.solve_one_step()
            except:
                self.assign_u0_to_u1()
                dt_val = 0.7*dt_val

                if (trial_count := trial_count + 1) >= max_trials:
                    raise RuntimeError('Reached max trial count. Abort!')

                continue

            trial_count = 0

            self.solve_solvent_amount(self.get_solution())

            self._solve_chem_equi_over_dofs()
            self.assign_u1_to_u0()

            self.solve_flow(target_residual=1e-10, max_steps=20)
            self.solve_auxiliary_variables()

            timestep += 1
            current_time += dt_val

            # Determine dt of the next time step.
            # TODO: Simplify dt determination to a function.

            if (dt_val := dt_val*1.1) > max_dt:
                dt_val = max_dt

            if dt_val < min_dt:
                dt_val = min_dt

            if (current_time + dt_val) > time_stamp:
                dt_val = time_stamp - current_time

                if time_stamps != []:
                    time_stamp = time_stamps.pop(0)

            if timestep%1 == 0:
                self.save_to_file(time=current_time)
                saved_times.append(current_time)

        if self.__MPI_rank==0:
            np.save(self.output_file_name, np.array(saved_times), allow_pickle=False)
