# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import reaktoro as rkt
from numpy import zeros, log, exp, array, arange

class ReactionManager:
    """
    Defines the default behavior of solving equilibrium reaction over dofs.
    """

    def setup_reaction_solver(self, temp=298.15):
        self.initialize_Reaktoro()

        self._set_temperature(temp, 'K') # Isothermal problem

        self.H_idx = self.component_dict['H+']

        num_dof = self.get_num_dof_per_component()
        self.rho_temp = zeros(num_dof)
        self.pH_temp = zeros(num_dof)
        self.lna_temp = zeros([num_dof, self.num_component+1])
        self.molar_density_temp = zeros([num_dof, self.num_component+1])

        self.set_dof_idx()

    def set_dof_idx(self):
        """Sets the dof indices to loop and perform equilibrium calculations."""

        num_dof = self.get_num_dof_per_component()
        self.dof_idx = arange(num_dof)

    def _solve_chem_equi_over_dofs(self, pressure, fluid_comp):
        for i in self.dof_idx:
            self._set_pressure(pressure[i], 'Pa')
            self._set_species_amount(list(fluid_comp[i]) + [self.solvent.vector()[i]])
            self.solve_chemical_equilibrium()

            self.rho_temp[i] = self._get_fluid_density()
            self.pH_temp[i] = self._get_fluid_pH()
            self.lna_temp[i] = self._get_species_log_activity_coeffs()
            self.molar_density_temp[i] = self._get_species_amounts()

    def _assign_chem_equi_results(self):
        self.fluid_density.vector()[:] = self.rho_temp*1e-6  #kg/m3 -> g/mm3
        self.fluid_pH.vector()[:] = self.pH_temp
        self.ln_activity.vector()[:] = self.lna_temp[:, :-1].flatten()

        self.fluid_components.vector()[:] = self.molar_density_temp[:, :-1].flatten()
        self.solvent.vector()[:] = self.molar_density_temp[:, -1].flatten()

    def initialize_Reaktoro(self):
        """
        """

        db = rkt.SupcrtDatabase("supcrt07")

        aqueous_components = self.component_str + ' ' + self.solvent_name

        self.aqueous_phase = rkt.AqueousPhase(aqueous_components)
        self.chem_system = rkt.ChemicalSystem(db, self.aqueous_phase)

        self.set_activity_models()

        self.chem_equi_solver = rkt.EquilibriumSolver(self.chem_system)

        self.chem_state = rkt.ChemicalState(self.chem_system)
        self.chem_prop = rkt.ChemicalProps(self.chem_state)
        self.aqueous_prop = rkt.AqueousProps(self.chem_state)

        self.one_over_ln10 = 1.0/log(10.0)

    def set_activity_models(self):
        self.aqueous_phase.set(rkt.ActivityModelHKF())

    def _set_temperature(self, value=298.0, unit='K'):
        self.chem_temp = value
        self.chem_state.setTemperature(value, unit)

    def _set_pressure(self, value=1.0, unit='atm'):
        self.chem_pres = value
        self.chem_state.setPressure(value, unit)

    def _set_species_amount(self, moles: list):
        self.chem_state.setSpeciesAmounts(moles)

    def solve_chemical_equilibrium(self):
        self.chem_equi_solver.solve(self.chem_state)
        self.chem_prop.update(self.chem_state)
        self.aqueous_prop.update(self.chem_state)

    def _get_species_amounts(self):
        return self.chem_state.speciesAmounts().asarray()

    def _get_charge_amount(self):
        return self.chem_state.elementAmounts()

    def _get_fluid_density(self):
        """The unit of density is kg/m3."""
        return self.chem_prop.density().val()

    def _get_fluid_pH(self):
        return -self.aqueous_prop.pH().val()

    def _get_fluid_volume(self):
        """In units of cubic meters."""
        return self.chem_prop.volume()