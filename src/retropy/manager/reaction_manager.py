# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import reaktoro as rkt
from numpy import zeros, log, arange

class ReactionManager:
    """
    Defines the default behavior of solving equilibrium reaction over dofs.
    """

    def setup_reaction_solver(self, temp=298.15):
        self.initialize_Reaktoro()
        self._set_temperature(temp, 'K') # Isothermal problem
        self.H_idx = self.component_dict['H+']
        num_dof = self.get_num_dof_per_component()

        self.initialize_fluid_pH()
        self.rho_temp = zeros(num_dof)
        self.pH_temp = zeros(num_dof)
        self.molar_density_temp = zeros([num_dof, self.chem_sys_dof])

        self.set_dof_idx()

    def set_dof_idx(self):
        """Sets the dof indices to loop and perform equilibrium calculations."""

        num_dof = self.get_num_dof_per_component()
        self.dof_idx = arange(num_dof)

    def _solve_chem_equi_over_dofs(self, pressure, fluid_comp):
        for i in self.dof_idx:
            self._set_pressure(pressure[i], 'Pa')
            self._set_species_amount(list(fluid_comp[i]) + [self.solvent.x.array[i]])
            self.solve_chemical_equilibrium()

            self.rho_temp[i] = self._get_fluid_density()
            self.pH_temp[i] = self._get_fluid_pH()
            self.molar_density_temp[i] = self._get_species_amounts()

    def _assign_chem_equi_results(self):
        idx_diff = self.num_component - self.chem_sys_dof
        self.fluid_density.x.array[:] = self.rho_temp * 1e-6  #kg/m3 -> g/mm3
        self.fluid_pH.x.array[:] = self.pH_temp
        self.fluid_components.x.array[:] = self.molar_density_temp[:, :idx_diff].flatten()
        self.solvent.x.array[:] = self.molar_density_temp[:, self.solvent_idx].flatten()

        self.fluid_density.x.scatter_forward()
        self.fluid_pH.x.scatter_forward()
        self.fluid_components.x.scatter_forward()
        self.solvent.x.scatter_forward()

    def initialize_Reaktoro(self, database="supcrt07"):
        """
        """

        self.set_chem_system(database)
        self.set_activity_models()

        self.chem_equi_solver = rkt.EquilibriumSolver(self.chem_system)

        self.chem_state = rkt.ChemicalState(self.chem_system)
        self.chem_prop = rkt.ChemicalProps(self.chem_state)
        self.aqueous_prop = rkt.AqueousProps(self.chem_state)

        self.one_over_ln10 = 1.0/log(10.0)

    def set_chem_system(self, database):
        db = rkt.SupcrtDatabase(database)
        aqueous_components = self.component_str + ' ' + self.solvent_name

        self.aqueous_phase = rkt.AqueousPhase(aqueous_components)
        self.chem_system = rkt.ChemicalSystem(db, self.aqueous_phase)
        self.chem_sys_dof = self.chem_system.species().size()
        self.solvent_idx = self.chem_system.species().index(self.solvent_name)

    def set_activity_models(self):
        self.aqueous_phase.set(rkt.ActivityModelHKF())

    def _set_temperature(self, value=298.15, unit='K'):
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

    def _get_fluid_density(self):
        """The unit of density is kg/m3."""
        return self.chem_prop.phaseProps("AqueousPhase").density().val()

    def _get_fluid_pH(self):
        return self.aqueous_prop.pH().val()

    def _get_fluid_volume(self):
        """In units of cubic meters."""
        return self.chem_prop.volume()