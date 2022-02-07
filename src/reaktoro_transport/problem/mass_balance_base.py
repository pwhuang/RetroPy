from . import *
import reaktoro as rkt
from numpy import array, log
from warnings import warn

class MassBalanceBase:
    """Base class for mass balance problems"""

    def set_components(self, *args):
        """
        Sets up the component dictionary.

        Input example: 'Na+', 'Cl-'
        """

        self.component_dict = {comp: idx for idx, comp in enumerate(args)}
        self.num_component = len(self.component_dict)

    def set_solvent(self, solvent='H2O(l)'):
        """The solvent is not included in transport calculations."""

        self.solvent_name = solvent

    def set_solvent_molar_mass(self, solvent_molar_mass=18.0e-3):
        self.M_solvent = solvent_molar_mass

    def set_solvent_ic(self, init_expr: Expression):
        self.solvent = interpolate(init_expr, self.DG0_space)
        self.solvent.rename(self.solvent_name, 'solvent')
        self._M_fraction = self._M/self.M_solvent

    def initiaize_ln_activity(self):
        self.ln_activity = Function(self.comp_func_spaces)
        self.ln_activity_dict = {}

        for comp_name, idx in self.component_dict.items():
            self.ln_activity_dict[f'lna_{comp_name}'] = idx

    def initialize_fluid_pH(self):
        self.fluid_pH = Function(self.DG0_space)
        self.fluid_pH.rename('pH', 'fluid_pH')

    def initialize_Reaktoro(self, database='supcrt07.xml'):
        """
        """

        editor = rkt.ChemicalEditor(rkt.Database(database))
        aqueous_phase = editor.addAqueousPhase(list(self.component_dict.keys()) + [self.solvent_name])
        aqueous_phase.setChemicalModelHKF()

        #TODO: write an interface for setting activity models
        # db = rkt.DebyeHuckelParams()
        # db.setPHREEQC()
        #
        # aqueous_phase.setChemicalModelDebyeHuckel(db)
        # aqueous_phase.setChemicalModelPitzerHMW()
        # aqueous_phase.setChemicalModelIdeal()

        self.chem_system = rkt.ChemicalSystem(editor)
        self.num_chem_elements = self.chem_system.numElements()
        self.__zeros = array([0.0]*self.num_chem_elements)

        self.chem_problem = rkt.EquilibriumProblem(self.chem_system)
        self.chem_equi_solver = rkt.EquilibriumSolver(self.chem_system)

        self.chem_state = rkt.ChemicalState(self.chem_system)
        self.chem_quant = rkt.ChemicalQuantity(self.chem_state)
        self.chem_prop = rkt.ChemicalProperties(self.chem_system)

        self.one_over_ln10 = 1.0/log(10.0)

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

    def _get_fluid_pH(self, idx):
        """The input idx should be the id of H+."""
        return -self.chem_prop.lnActivities().val[idx]*self.one_over_ln10

    def _get_fluid_volume(self):
        """In units of cubic meters."""
        return self.chem_prop.fluidVolume().val
