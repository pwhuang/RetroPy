import os
os.environ['OMP_NUM_THREADS'] = '1'

from mesh_factory import MeshFactory
from reaktoro_transport.manager import DarcyFlowManagerUzawa as FlowManager
from reaktoro_transport.manager import TransportManager, ReactionManager

import numpy as np
from dolfin import (info, end, Expression, MPI, Constant)

class Problem(FlowManager, TransportManager, ReactionManager,
              MeshFactory):
    """This class solves the chemically driven convection problem."""

    def __init__(self, nx, ny, const_diff):
        TransportManager.__init__(self, *self.get_mesh_and_markers(nx, ny), const_diff)
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

        self.H_idx = 2
        self.background_pressure = 101325 + 1e-3*9806.65*25 # Pa

        HCl_amounts = [1e-13, 1.0, 1.0, 1e-13, 54.17] # micro mol/mm^3 # mol/L
        NaOH_amounts = [1.0, 1e-13, 1e-13, 1.0, 55.36]

        init_expr_list = []

        for i in range(self.num_component):
            init_expr_list.append('x[1]<=25.0 ?' + str(NaOH_amounts[i]) + ':' + str(HCl_amounts[i]))

        self.set_component_ics(Expression(init_expr_list, degree=1))
        self.set_solvent_ic(Expression('x[1]<=25.0 ?' + str(NaOH_amounts[-1]) + ':' + str(HCl_amounts[-1]) , degree=1))

    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity(8.9e-4)  # Pa sec
        self.set_gravity([0.0, -9806.65]) # mm/sec
        self.set_permeability(0.5**2/12.0) # mm^2

    def set_flow_ibc(self):
        self.mark_flow_boundary(pressure = [],
                                velocity = [self.marker_dict['top'], self.marker_dict['bottom'],
                                            self.marker_dict['left'], self.marker_dict['right']])

        self.set_pressure_bc([]) # Pa
        self.set_pressure_ic(Constant(0.0))
        self.set_velocity_bc([Constant([0.0, 0.0])]*4)

    def solve_species_transport(self):
        max_trials = 7

        try:
            self.solve_one_step()
            is_solved = True
        except:
            self.assign_u0_to_u1()

            if self.trial_count >= max_trials:
                raise RuntimeError('Reached max trial count. Abort!')
            end() # Added to avoid unbalanced indentation in logs.
            is_solved = False

        return is_solved

    @staticmethod
    def timestepper(dt_val, current_time, time_stamp):
        min_dt, max_dt = 5e-2, 2.0

        if (dt_val := dt_val*1.1) > max_dt:
            dt_val = max_dt
        elif dt_val < min_dt:
            dt_val = min_dt
        if dt_val > time_stamp - current_time:
            dt_val = time_stamp - current_time

        return dt_val

    def solve_initial_condition(self):
        self.assign_u0_to_u1()
        self._solve_chem_equi_over_dofs()
        self.solve_flow(target_residual=1e-10, max_steps=50)

    def save_to_file(self, time):
        super().save_to_file(time, is_saving_pv=True)
        self._save_function(time, self.solvent)
        self._save_function(time, self.fluid_density)
        self._save_function(time, self.fluid_pH)
        self._save_mixed_function(time, self.ln_activity, self.ln_activity_dict)

    def solve(self, dt_val=1.0, endtime=10.0, time_stamps=[]):
        self.solve_initial_condition()

        current_time = 0.0
        timestep = 1
        saved_times = []
        self.trial_count = 0

        time_stamps.append(endtime)
        time_stamp_idx = 0
        time_stamp = time_stamps[time_stamp_idx]

        self.save_to_file(time=current_time)
        saved_times.append(current_time)
        save_interval = 1

        while current_time < endtime:
            if self.__MPI_rank==0:
                info(f"timestep = {timestep}, dt = {dt_val:.6f}, "\
                     f"current_time = {current_time:.6f}\n")

            self.set_dt(dt_val)

            if self.solve_species_transport() is False:
                dt_val = 0.7*dt_val
                self.trial_count += 1
                continue

            self.trial_count = 0
            self.solve_solvent_transport()
            self._solve_chem_equi_over_dofs()
            self.assign_u1_to_u0()

            self.solve_flow(target_residual=1e-10, max_steps=20)

            timestep += 1
            if (current_time := current_time + dt_val) >= time_stamp:
                time_stamp_idx += 1
                try:
                    time_stamp = time_stamps[time_stamp_idx]
                except:
                    time_stamp = time_stamps[-1]

            dt_val = self.timestepper(dt_val, current_time, time_stamp)

            if timestep % save_interval == 0:
                self.save_to_file(time=current_time)
                saved_times.append(current_time)

        if self.__MPI_rank==0:
            np.save(self.output_file_name, np.array(saved_times), allow_pickle=False)
