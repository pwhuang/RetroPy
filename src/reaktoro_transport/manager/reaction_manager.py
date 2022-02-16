import reaktoro as rkt
from numpy import zeros, log, exp, array, arange
from warnings import warn

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

            self.rho_temp[i] = self._get_fluid_density()*1e-6  #g/mm3
            self.pH_temp[i] = self._get_fluid_pH()
            self.lna_temp[i] = self._get_species_log_activity_coeffs()
            self.molar_density_temp[i] = self._get_species_amounts()

    def _assign_chem_equi_results(self):
        self.fluid_density.vector()[:] = self.rho_temp
        self.fluid_pH.vector()[:] = self.pH_temp
        self.ln_activity.vector()[:] = self.lna_temp[:, :-1].flatten()

        self.fluid_components.vector()[:] = self.molar_density_temp[:, :-1].flatten()
        self.solvent.vector()[:] = self.molar_density_temp[:, -1].flatten()

    def initialize_Reaktoro(self, database='supcrt07.xml'):
        """
        """

        editor = self.set_chem_editor(database)
        self.set_activity_models()

        self.chem_system = rkt.ChemicalSystem(editor)
        self.num_chem_elements = self.chem_system.numElements()
        self.__zeros = array([0.0]*self.num_chem_elements)

        self.chem_problem = rkt.EquilibriumProblem(self.chem_system)
        self.chem_equi_solver = rkt.EquilibriumSolver(self.chem_system)

        self.chem_state = rkt.ChemicalState(self.chem_system)
        self.chem_quant = rkt.ChemicalQuantity(self.chem_state)
        self.chem_prop = rkt.ChemicalProperties(self.chem_system)

        self.one_over_ln10 = 1.0/log(10.0)

    def set_activity_models(self):
        self.aqueous_phase.setChemicalModelHKF()

    def set_chem_editor(self, database):
        editor = rkt.ChemicalEditor(rkt.Database(database))
        self.aqueous_phase = editor.addAqueousPhase(list(self.component_dict.keys()) + [self.solvent_name])

        return editor

    def set_smart_equilibrium_solver(self, reltol=1e-3, amount_fraction_cutoff=1e-14,
                                     mole_fraction_cutoff=1e-14):

        try:
            rkt.SmartEquilibriumOptions()
        except:
            warn("\nThe installed Reaktoro version does not support"
                 "SmartEquilibriumSolver! EquilibriumSolver is used.")
            return

        self.chem_equi_solver = rkt.SmartEquilibriumSolver(self.chem_system)
        smart_equi_options = rkt.SmartEquilibriumOptions()

        smart_equi_options.reltol = reltol
        smart_equi_options.amount_fraction_cutoff = amount_fraction_cutoff
        smart_equi_options.mole_fraction_cutoff = mole_fraction_cutoff

        self.chem_equi_solver = rkt.SmartEquilibriumSolver(self.chem_system)
        self.chem_equi_solver.setOptions(smart_equi_options)

    def _set_temperature(self, value=298.0, unit='K'):
        self.chem_temp = value
        self.chem_problem.setTemperature(value, unit)

    def _set_pressure(self, value=1.0, unit='atm'):
        self.chem_pres = value
        self.chem_problem.setPressure(value, unit)

    def _set_species_amount(self, moles: list):
        self.chem_state.setSpeciesAmounts(moles)
        self.chem_problem.setElementAmounts(self.__zeros)
        self.chem_problem.addState(self.chem_state)
        self.chem_problem.setElectricalCharge(1e-16)

    def solve_chemical_equilibrium(self):
        self.chem_equi_solver.solve(self.chem_state, self.chem_problem)
        self.chem_prop.update(self.chem_temp, self.chem_pres,\
                              self._get_species_amounts())

    def _get_species_amounts(self):
        return self.chem_state.speciesAmounts()

    def _get_charge_amount(self):
        return self.chem_state.elementAmount('Z')

    def _get_species_log_activity_coeffs(self):
        return self.chem_prop.lnActivityCoefficients().val

    def _get_species_chemical_potentials(self):
        """The unit of the chemical potential is J/mol"""
        return self.chem_prop.chemicalPotentials().val

    def _get_fluid_density(self):
        """The unit of density is kg/m3."""
        return self.chem_prop.phaseDensities().val[0]

    def _get_fluid_pH(self):
        return -self.chem_prop.lnActivities().val[self.H_idx]*self.one_over_ln10

    def _get_fluid_volume(self):
        """In units of cubic meters."""
        return self.chem_prop.fluidVolume().val
