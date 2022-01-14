import sys
sys.path.insert(0, '../../')

from numpy import zeros, log, array, exp

class ReactionManager:
    def setup_reaction_solver(self):
        self.initialize_Reaktoro()
        self.set_smart_equilibrium_solver()

        self._set_temperature(298, 'K') # Isothermal problem
        self.initiaize_ln_activity()

        self.num_dof = self.get_num_dof_per_component()
        self.rho_temp = zeros(self.num_dof)
        self.lna_temp = zeros([self.num_dof, self.num_component+1])
        self.molar_density_temp = zeros([self.num_dof, self.num_component+1])

    def _solve_chem_equi_over_dofs(self):
        pressure = self.fluid_pressure.vector()[:]
        fluid_comp = self.get_solution()
        component_molar_density = exp(fluid_comp.vector()[:].reshape(-1, self.num_component))

        molar_density_list = []
        rho_list = []

        for i in range(self.num_dof):
            self._set_pressure(pressure[i] + self.background_pressure, 'Pa')
            self._set_species_amount(list(component_molar_density[i]) + [self.solvent.vector()[i]])
            self.solve_chemical_equilibrium()

            self.rho_temp[i] = self._get_fluid_density()*1e-6  #g/mm3
            self.lna_temp[i] = self._get_species_log_activity_coeffs()
            self.molar_density_temp[i] = self._get_species_amounts()

        self.fluid_density.vector()[:] = self.rho_temp.flatten()
        self.ln_activity.vector()[:] = self.lna_temp[:, :-1].flatten()
        fluid_comp.vector()[:] = log(self.molar_density_temp[:, :-1].flatten())
        self.solvent.vector()[:] = self.molar_density_temp[:,-1].flatten()

    def solve_solvent_amount(self, fluid_comp_new):
        self.solvent.vector()[:] += \
        ((exp(self.fluid_components.vector()) - exp(fluid_comp_new.vector())).reshape(-1, self.num_component)\
        *self._M_fraction).sum(axis=1)

        #print(self.solvent.vector().sum())
