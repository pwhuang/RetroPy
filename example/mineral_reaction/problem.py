# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from mesh_factory import MeshFactory
from retropy.problem import DarcyFlowMixedPoisson
from retropy.manager import ReactiveTransportManager
from retropy.manager import XDMFManager as OutputManager
from retropy.solver import TransientNLSolver

from dolfinx.fem import Constant, form, Function
from dolfinx.fem.petsc import assemble_vector
from ufl import as_vector

import reaktoro as rkt
import numpy as np

class FlowManager(DarcyFlowMixedPoisson):
    def setup_flow_solver(self, r_val, omega_by_r):
        self.set_flow_fe_space()
        self.set_fluid_properties()

        self.generate_form()
        self.generate_residual_form()
        self.set_flow_ibc()

        self.set_additional_parameters(r_val=r_val, omega_by_r=omega_by_r)
        self.assemble_matrix()

        solver_params = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
        }

        self.set_flow_solver_params(solver_params)

class ReactiveTransportManager(ReactiveTransportManager):
    def __init__(self, marked_mesh):
        super().__init__(marked_mesh)
        self.interpolated_velocity = Function(self.Vec_DG0_space)
        self.interpolated_velocity.name = 'fluid velocity'

    def add_physics_to_form(self, u, kappa, f_id=0):
        theta_val = 1.0

        theta = Constant(self.mesh, theta_val)
        one = Constant(self.mesh, 1.0)

        # self.add_explicit_advection(u, kappa=one, marker=0, f_id=f_id)
        self.add_implicit_advection(kappa=one, marker=0, f_id=f_id)

        for component in self.component_dict.keys():
            self.add_implicit_diffusion(component, kappa=theta, marker=0)
            self.add_explicit_diffusion(component, u, kappa=one-theta, marker=0)

        if self.is_same_diffusivity==False:
            self.add_semi_implicit_charge_balanced_diffusion(u, kappa=theta, marker=0)
            self.add_explicit_charge_balanced_diffusion(u, kappa=one-theta, marker=0)

    def set_chem_system(self, database):
        db = rkt.PhreeqcDatabase('pitzer.dat')
        aqueous_components = self.component_str + ' ' + self.solvent_name

        self.aqueous_phase = rkt.AqueousPhase(aqueous_components)
        self.mineral_phase = rkt.MineralPhase(self.mineral_name)
        self.chem_system = rkt.ChemicalSystem(db, self.aqueous_phase, self.mineral_phase)
        
        self.chem_sys_dof = self.chem_system.species().size()
        self.solvent_idx = self.chem_system.species().index(self.solvent_name)

    def set_activity_models(self):
        self.aqueous_phase.set(rkt.chain(rkt.ActivityModelPitzer()))

    def set_advection_velocity(self):
        self.advection_velocity = as_vector(
            [self.fluid_velocity / self._phi for _ in range(self.num_component)]
        )

    def setup_reaction_solver(self, temp=298.15):
        super().setup_reaction_solver()
        num_dof = self.get_num_dof_per_component()
        self.molar_density_temp = np.zeros([num_dof, self.num_component+2])

    def _solve_chem_equi_over_dofs(self, pressure, fluid_comp):
        for i in self.dof_idx:
            self._set_pressure(pressure[i], 'Pa')
            self._set_species_amount(list(fluid_comp[i]) + [self.solvent.x.array[i]] + [self.mineral.x.array[i]])
            self.solve_chemical_equilibrium()

            self.rho_temp[i] = self._get_fluid_density()
            self.pH_temp[i] = self._get_fluid_pH()
            self.molar_density_temp[i] = self._get_species_amounts()

    def _assign_chem_equi_results(self):
        super()._assign_chem_equi_results()
        self.mineral.x.array[:] = self.molar_density_temp[:, -1].flatten()

    def save_to_file(self, time):
        super().save_to_file(time)
        self.write_function(self.mineral, time)
        self.interpolated_velocity.interpolate(self.fluid_velocity)
        self.write_function(self.interpolated_velocity, time)
        self.write_function(self.fluid_pressure, time)

class FlowManager(FlowManager):
    def set_flow_fe_space(self):
        self.set_pressure_fe_space('DG', 0)

        if self.mesh.topology.cell_name() == 'triangle':
            self.set_velocity_fe_space('RT', 1)
        elif self.mesh.topology.cell_name() == 'quadrilateral':
            self.set_velocity_fe_space('RTCF', 1)

    def set_flow_ibc(self):
        """Sets the initial and boundary conditions of the flow."""

        self.set_pressure_ic(lambda x: 0.0 * x[0])
        self.set_pressure_bc({'right': Constant(self.mesh, 101325.)})
        self.add_weak_pressure_bc(penalty_value=20.0)

        velocity_bc = Function(self.velocity_func_space)
        velocity_bc.x.array[:] = 0.0
        velocity_bc.x.scatter_forward()

        darcy_flux = 1.98e-2  ## mm/sec

        left_inlet_bc = Function(self.velocity_func_space)
        left_inlet_bc.interpolate(lambda x: (darcy_flux + 0.0*x[0], 0.0*x[1]))

        self.set_velocity_bc({'top': velocity_bc,
                              'left': left_inlet_bc,
                              'bottom': velocity_bc,
                             })

class Problem(ReactiveTransportManager, FlowManager, OutputManager,
              TransientNLSolver):
    """This class solves the CO2 convection problem."""

    def __init__(self, nx, ny, const_diff):
        super().__init__(MeshFactory(nx, ny))
        self.is_same_diffusivity = const_diff
        self.set_flow_residual(5e-10)

    def set_component_properties(self):
        self.set_molar_mass([1.00794, 17.00734, 22.9898, 35.453, 137.327, 96.06]) #g/mol
        self.set_solvent_molar_mass(18.0153)
        self.set_charge([1.0, -1.0, 1.0, -1.0, 2.0, -2.0])

    def define_problem(self):
        self.set_components('H+ OH- Ca+2 HCO3- CO3-2 CO2')
        self.set_solvent('H2O')
        self.set_mineral('Calcite')
        self.set_component_properties()

        self.set_component_fe_space()
        self.initialize_form()

        self.background_pressure = 101325. # Pa, 1 atm
        self.injected_amount = 0.01 # mol / L
        init_conc = [1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 55.336, 1e-16]

        for comp, concentration in zip(self.component_dict.keys(), init_conc):
            self.set_component_ics(comp, lambda x: 0.0 * x[0] + concentration)

        self.set_solvent_ic(lambda x: 0.0 * x[0] + init_conc[-2])
        self.set_mineral_ic(lambda x: 0.0 * x[0] + 1.0)

        self.mark_component_boundary(
            {
                "H+": [self.marker_dict["left"]],
                "OH-": [self.marker_dict["left"]],
                "Ca+2": [self.marker_dict["left"]],
                "HCO3-": [self.marker_dict["left"]],
                "CO3-2": [self.marker_dict["left"]],
                "CO2": [self.marker_dict["left"]],
                "outlet": [self.marker_dict["right"]],
            }
        )

    def add_physics_to_form(self, u, kappa, f_id=0):
        super().add_physics_to_form(u, kappa, f_id)

        C_co2 = self.injected_amount
        
        inlet_conc = [1e-16, 1e-16, 1e-16, 1e-16, 1e-16, C_co2] # micro mol/mm^3 # mol/L

        for comp, conc in zip(self.component_dict, inlet_conc):
            self.add_component_advection_bc(comp, [conc])
        
        self.add_outflow_bc(f_id)

    def set_fluid_properties(self):
        self.set_porosity(0.2)
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity(0.893e-3)  # Pa sec
        self.set_gravity([9806.65, 0.0]) # mm/sec^2
        self.set_permeability(0.01 / 12.) # mm^2

    def update_permeability(self):
        # TODO: define permeability based on local cubic law or Kozeny-Carman equation
        pass

    @staticmethod
    def timestepper(dt_val, current_time, time_stamp):
        min_dt, max_dt = 5e-3, 10.0

        if (dt_val := dt_val*1.1) > max_dt:
            dt_val = max_dt
        elif dt_val < min_dt:
            dt_val = min_dt
        if dt_val > time_stamp - current_time:
            dt_val = time_stamp - current_time

        return dt_val
    
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

        saved_times.append(current_time)
        flow_residuals.append(self.get_flow_residual())
        save_interval = 1
        flush_interval = 25

        while current_time < endtime:
            if self.__MPI_rank==0:
                print(f"timestep = {timestep}, dt = {dt_val:.6f}, "\
                      f"current_time = {current_time:.6f}\n")

            self.dt.value = dt_val

            newton_steps, is_solved = self.solve_species_transport()

            if is_solved is False:
                dt_val = 0.7*dt_val
                self.trial_count += 1
                continue

            if self.__MPI_rank==0:
                print(f"Transport solve converged. Newton steps = {newton_steps}.\n")

            self.trial_count = 0
            self.solve_solvent_transport()

            fluid_comp = np.exp(self.get_solver_u1().x.array.reshape(-1, self.num_component))
            pressure = self.fluid_pressure.x.array + self.background_pressure
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

        if self.__MPI_rank==0:
            np.save(self.output_file_name + '_time', np.array(saved_times), allow_pickle=False)
            np.save(self.output_file_name + '_flow_res', np.array(flow_residuals), allow_pickle=False)
