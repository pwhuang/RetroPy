import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys

from mesh_factory import MeshFactory
from flow_manager_uzawa import FlowManager
from transport_manager import TransportManager

from numpy import shape, array
from dolfin import info, DOLFIN_EPS, assemble

class main(FlowManager, TransportManager, MeshFactory):
    """
    This class solves the chemically driven convection problem.
    """

    def __init__(self, nx, ny):
        TransportManager.__init__(self, *self.get_mesh_and_markers(nx, ny))

    def _solve_chem_equi_over_dofs(self):
        pressure = self.fluid_pressure.vector()[:]
        component_molar_density = self.fluid_components.vector()[:].reshape(-1, self.num_component)
        solvent_molar_density = self.solvent.vector()[:]

        molar_density_list = []
        rho_list = []

        for i in range(self.num_dof):
            self._set_pressure(pressure[i] + self.background_pressure, 'Pa')
            self._set_species_amount(list(component_molar_density[i]) + [solvent_molar_density[i]])
            super().solve_chemical_equilibrium()
            self.molar_density_temp[i] = self._get_species_amounts()
            self.rho_temp[i] = self._get_fluid_density()*1e-6

        self.fluid_components.vector()[:] = self.molar_density_temp[:, :-1].flatten()
        self.fluid_density.vector()[:] = self.rho_temp

    def solve_initial_condition(self):
        self._solve_chem_equi_over_dofs()
        self._rho_old.assign(self.fluid_density)

    def solve_chemical_equilibrium(self):
        self._solve_solvent_amount(self.get_solution())
        self._solve_chem_equi_over_dofs()

    def save_fluid_density(self, time):
        self.xdmf_obj.write_checkpoint(self.fluid_density,
                                       self.fluid_density.name(),
                                       time_step=time,
                                       append=True)

    def save_to_file(self, time):
        if not super().save_to_file(time):
            return False

        self.save_fluid_density(time)
        return True

    def timestepper(self):
        pass

    def solve(self, dt_val=1.0, endtime=10.0):
        self.solve_initial_condition()
        self.solve_flow(target_residual=1e-13, max_steps=100)

        current_time = 0.0
        timestep = 1
        self.save_to_file(time=current_time)

        while current_time < endtime:
            info('timestep = ' + str(timestep) + ',  dt = ' + str(dt_val)\
                 + ', current_time = ' + str(current_time) )

            self.set_dt(dt_val)
            self.solve_one_step()

            if (min_val := self.get_solution().vector().min()) < -1e-6:
                info('min_val = ' + str(min_val))
                dt_val = dt_val*0.67
                continue

            self.assign_u1_to_u0()
            self.solve_chemical_equilibrium()

            # print(assemble((self._rho_old - self.fluid_density)*\
            #                (self._rho_old - self.fluid_density)/self.dt*self.dx))

            self.solve_flow(target_residual=5e-10, max_steps=25)

            self._rho_old.assign(self.fluid_density)

            timestep += 1
            current_time += dt_val

            dt_max = 0.1

            if (dt_val := dt_val*1.2) > dt_max:
                dt_val = dt_max

            self.save_to_file(time=current_time)


problem = main(nx=16, ny=26)
problem.generate_output_instance(sys.argv[1])
problem.define_problem()
problem.setup_flow_solver()
problem.setup_transport_solver()
problem.solve(dt_val=1e-3, endtime=100.0)
