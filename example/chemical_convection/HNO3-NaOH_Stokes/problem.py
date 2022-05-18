import os
os.environ['OMP_NUM_THREADS'] = '1'

from mesh_factory import MeshFactory
from reaktoro_transport.manager import StokesFlowManagerUzawa as FlowManager
from reaktoro_transport.manager import ReactiveTransportManager
from reaktoro_transport.manager import XDMFManager as OutputManager

from reaktoro_transport.solver import CustomNLSolver

from dolfin import (Expression, Constant, PETScOptions, interpolate, Function,
                    project, info, end)
import numpy as np

class ReactiveTransportManager(ReactiveTransportManager):
    def add_physics_to_form(self, u, theta_val=0.5, f_id=0):
        """
        Crank-Nicolson advection and diffusion.
        """

        self.set_advection_velocity()

        theta = Constant(theta_val)
        one = Constant(1.0)

        self.add_implicit_advection(kappa=theta, marker=0, f_id=f_id)
        self.add_explicit_advection(u, kappa=one-theta, marker=0, f_id=f_id)

        for component in self.component_dict.keys():
            self.add_implicit_diffusion(component, kappa=theta, marker=0)
            self.add_explicit_diffusion(component, u, kappa=one-theta, marker=0)

        if self.is_same_diffusivity==False:
            self.add_implicit_charge_balanced_diffusion(kappa=theta, marker=0)
            self.add_explicit_charge_balanced_diffusion(u, kappa=one-theta, marker=0)

    def solve_initial_condition(self):
        self.assign_u0_to_u1()

        # updates the pressure assuming constant density
        self.solve_flow(target_residual=self.flow_residual, max_steps=50)

        self.fluid_pressure_DG = Function(self.DG0_space)

        fluid_comp = np.exp(self.get_solution().vector()[:].reshape(-1, self.num_component))

        self.fluid_pressure_DG.vector()[:] = interpolate(self.fluid_pressure, self.DG0_space).vector()[:]
        pressure = self.fluid_pressure_DG.vector()[:] + self.background_pressure
        self._solve_chem_equi_over_dofs(pressure, fluid_comp)
        self._assign_chem_equi_results()

        # updates the pressure and velocity using the density at equilibrium
        self.solve_flow(target_residual=self.flow_residual, max_steps=50)

    def solve_species_transport(self):
        max_trials = 7

        try:
            is_solved = self.solve_one_step()

            if is_solved is False:
                raise RuntimeError('SNES solve does not converge.')
        except:
            self.assign_u0_to_u1()

            if self.trial_count >= max_trials:
                raise RuntimeError('Reached max trial count. Abort!')
            end() # Added to avoid unbalanced indentation in logs.
            is_solved = False

        return is_solved

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
            self.fluid_pressure_DG.vector()[:] = interpolate(self.fluid_pressure, self.DG0_space).vector()[:]
            pressure = self.fluid_pressure_DG.vector()[:] + self.background_pressure
            self._solve_chem_equi_over_dofs(pressure, fluid_comp)
            self._assign_chem_equi_results()
            self.solve_flow(target_residual=self.flow_residual, max_steps=10)

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

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        opts = self.get_solver_parameters()

        #opts['snes_view'] = None
        opts['snes_monitor'] = None
        #opts['snes_linesearch_monitor'] = None
        opts['snes_converged_reason'] = None
        opts['snes_type'] = 'newtonls'
        opts['snes_linesearch_type'] = 'bt'
        opts['snes_linesearch_alpha'] = 2e-1
        opts['snes_linesearch_damping'] = 1.2
        opts['snes_linesearch_minlambda'] = 1e-10
        opts['snes_linesearch_setorder'] = 2

        #opts['ksp_monitor_true_residual'] = None
        opts['ksp_max_it'] = 500
        opts['ksp_rtol'] = 1e-12
        opts['ksp_atol'] = 1e-14
        #opts['ksp_converged_reason'] = None

        # opts['pc_hypre_boomeramg_strong_threshold'] = 0.25
        # opts['pc_hypre_boomeramg_truncfactor'] = 0.0
        # opts['pc_hypre_boomeramg_print_statistics'] = 0

        self.snes.setFromOptions()

        self.snes.getKSP().setInitialGuessNonzero(True)
        self.snes.setTolerances(rtol=1e-15, atol=1e-15, stol=1e-16, max_it=100)
        self.snes.getKSP().setType(linear_solver)
        self.snes.getKSP().getPC().setType(preconditioner)
        self.snes.getKSP()
        #self.snes.getKSP().getPC().setHYPREType('boomeramg')
        #self.snes.getKSP().getPC().setGAMGType('agg')

class Problem(ReactiveTransportManager, FlowManager, MeshFactory, OutputManager,
              CustomNLSolver):
    """This class solves the chemically driven convection problem."""

    def __init__(self, const_diff):
        super().__init__(*self.get_mesh_and_markers())
        self.is_same_diffusivity = const_diff

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

        self.background_pressure = 1e5 + 1e-3*9806.65*0.5 # Pa

        HNO3_amounts = [1e-15, 1.5, 1.5, 1e-13, 52.712] # micro mol/mm^3
        NaOH_amounts = [1.4, 1e-15, 1e-15, 1.4, 55.361]

        init_expr_list = []

        for i in range(self.num_component):
            init_expr_list.append('x[1]<=2.5 ?' + str(NaOH_amounts[i]) + ':' + str(HNO3_amounts[i]))

        self.set_component_ics(Expression(init_expr_list, degree=1))
        self.set_solvent_ic(Expression('x[1]<=2.5 ?' + str(NaOH_amounts[-1]) + ':' + str(HNO3_amounts[-1]) , degree=1))

    def set_fluid_properties(self):
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity((1.232e-3 + 0.933e-3)*0.5)  # Pa sec
        self.set_gravity([0.0, -9806.65]) # mm/sec

    def set_flow_ibc(self):
        self.mark_flow_boundary(inlet = [], noslip = [555], velocity_bc = [])

        self.set_pressure_bc([]) # Pa
        self.set_pressure_ic(Constant(0.0))
        self.set_velocity_bc([])

    def setup_transport_solver(self):
        self.generate_solver(eval_jacobian=True)
        self.set_solver_parameters('gmres', 'gamg')

    @staticmethod
    def timestepper(dt_val, current_time, time_stamp):
        min_dt, max_dt = 1e-3, 0.4

        if (dt_val := dt_val*1.05) > max_dt:
            dt_val = max_dt
        elif dt_val < min_dt:
            dt_val = min_dt
        if dt_val > time_stamp - current_time:
            dt_val = time_stamp - current_time

        return dt_val

    def set_flow_fe_space(self):
        self.set_pressure_fe_space('DG', 1)
        self.set_velocity_vector_fe_space('CG', 2)

    def save_fluid_velocity(self, time_step):
        #self.fluid_vel_to_save = interpolate(self.fluid_velocity, self.Vec_DG0_space)
        self.write_function(self.fluid_velocity, self.fluid_velocity.name(),
                            time_step)
