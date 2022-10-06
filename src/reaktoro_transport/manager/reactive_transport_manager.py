# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import ReactionManager, TransportManager
import numpy as np
from dolfin import info, end, MPI

class ReactiveTransportManager(TransportManager, ReactionManager):
    """Defines the default behavior of solving reactive transport problems."""

    def __init__(self, mesh, boundary_markers, domain_markers):
        super().__init__(mesh, boundary_markers, domain_markers)
        self.__MPI_rank = MPI.rank(MPI.comm_world)
        self.set_flow_residual(1e-10)

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
        min_dt, max_dt = 1e-2, 1.0

        dt_val = dt_val*1.1
        if dt_val > max_dt:
            dt_val = max_dt
        elif dt_val < min_dt:
            dt_val = min_dt
        if dt_val > time_stamp - current_time:
            dt_val = time_stamp - current_time

        return dt_val

    def set_flow_residual(self, residual):
        self.flow_residual = residual

    def solve_initial_condition(self):
        self.assign_u0_to_u1()

        # updates the pressure assuming constant density
        self.solve_flow(target_residual=self.flow_residual, max_steps=50)

        fluid_comp = np.exp(self.get_solution().vector()[:].reshape(-1, self.num_component))
        pressure = self.fluid_pressure.vector()[:] + self.background_pressure
        self._solve_chem_equi_over_dofs(pressure, fluid_comp)
        self._assign_chem_equi_results()

        # updates the pressure and velocity using the density at equilibrium
        self.solve_flow(target_residual=self.flow_residual, max_steps=50)

    def save_to_file(self, time):
        super().save_to_file(time, is_exponentiated=True, is_saving_pv=True)
        self.write_function(self.solvent, self.solvent.name(), time)
        self.write_function(self.fluid_density, self.fluid_density.name(), time)
        self.write_function(self.fluid_pH, self.fluid_pH.name(), time)
        # self._save_mixed_function(time, self.ln_activity, self.ln_activity_dict)
        if self.__MPI_rank==0:
            self.csv_writer.writerow([time])

    def solve(self, dt_val=1.0, endtime=10.0, time_stamps=[]):
        current_time = 0.0
        timestep = 1
        saved_times = []
        flow_residuals = []
        self.trial_count = 0

        time_stamps.append(endtime)
        time_stamp_idx = 0
        time_stamp = time_stamps[time_stamp_idx]

        self.solve_initial_condition()
        self.save_to_file(time=current_time)
        self.logarithm_fluid_components()

        saved_times.append(current_time)
        flow_residuals.append(self.get_flow_residual())
        save_interval = 1
        flush_interval = 25

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

            fluid_comp = np.exp(self.get_solution().vector()[:].reshape(-1, self.num_component))
            pressure = self.fluid_pressure.vector()[:] + self.background_pressure
            self._solve_chem_equi_over_dofs(pressure, fluid_comp)
            self._assign_chem_equi_results()
            self.solve_flow(target_residual=self.flow_residual, max_steps=20)

            timestep += 1

            current_time = current_time + dt_val
            if current_time >= time_stamp:
                time_stamp_idx += 1
                try:
                    time_stamp = time_stamps[time_stamp_idx]
                except:
                    time_stamp = time_stamps[-1]

            dt_val = self.timestepper(dt_val, current_time, time_stamp)

            if timestep % save_interval == 0:
                self.save_to_file(time=current_time)
                saved_times.append(current_time)
                flow_residuals.append(self.get_flow_residual())

            if timestep % flush_interval == 0:
                self.flush_output()

            self.logarithm_fluid_components()

        if self.__MPI_rank==0:
            np.save(self.output_file_name + '_time', np.array(saved_times), allow_pickle=False)
            np.save(self.output_file_name + '_flow_res', np.array(flow_residuals), allow_pickle=False)
