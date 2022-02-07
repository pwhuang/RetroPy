from numpy import zeros, log, exp

class ReactionManager:
    """
    Defines the default behavior of solving equilibrium reaction over dofs.
    """

    def setup_reaction_solver(self, temp=298.15):
        self.initialize_Reaktoro()
        self.set_smart_equilibrium_solver()

        self._set_temperature(temp, 'K') # Isothermal problem

        self.initiaize_ln_activity()
        self.initialize_fluid_pH()

        self.num_dof = self.get_num_dof_per_component()
        self.rho_temp = zeros(self.num_dof)
        self.pH_temp = zeros(self.num_dof)
        self.lna_temp = zeros([self.num_dof, self.num_component+1])
        self.molar_density_temp = zeros([self.num_dof, self.num_component+1])

    def _solve_chem_equi_over_dofs(self):
        pressure = self.fluid_pressure.vector()[:] + self.background_pressure
        fluid_comp = self.get_solution()
        component_molar_density = exp(fluid_comp.vector()[:].reshape(-1, self.num_component))

        for i in range(self.num_dof):
            self._set_pressure(pressure[i], 'Pa')
            self._set_species_amount(list(component_molar_density[i]) + [self.solvent.vector()[i]])
            self.solve_chemical_equilibrium()

            self.rho_temp[i] = self._get_fluid_density()*1e-6  #g/mm3
            self.pH_temp[i] = self._get_fluid_pH(self.H_idx)
            self.lna_temp[i] = self._get_species_log_activity_coeffs()
            self.molar_density_temp[i] = self._get_species_amounts()

        self.fluid_density.vector()[:] = self.rho_temp
        self.fluid_pH.vector()[:] = self.pH_temp
        self.ln_activity.vector()[:] = self.lna_temp[:, :-1].flatten()

        fluid_comp.vector()[:] = log(self.molar_density_temp[:, :-1].flatten())
        self.solvent.vector()[:] = self.molar_density_temp[:, -1].flatten()
