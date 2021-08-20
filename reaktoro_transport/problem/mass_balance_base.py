from . import *
import reaktoro as rkt

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
        self.solvent = solvent

    def initiaize_ln_activity(self):
        self.ln_activity = Function(self.comp_func_spaces)

    def initialize_Reaktoro(self, database='supcrt07.xml'):
        """
        """

        editor = rkt.ChemicalEditor(rkt.Database(database))
        editor.addAqueousPhase(list(self.component_dict.keys()) + [self.solvent])

        system = rkt.ChemicalSystem(editor)

        self.chem_equi_solver = rkt.EquilibriumSolver(system)
        self.chem_state = rkt.ChemicalState(system)
        self.chem_quant = rkt.ChemicalQuantity(self.chem_state)
        #self.chem_prop = rkt.ChemicalProperties(system)

    def _set_temperature(self, value=298.0, unit='K'):
        self.chem_state.setTemperature(value, unit)

    def _set_pressure(self, value=1.0, unit='atm'):
        self.chem_state.setPressure(value, unit)

    def _set_species_amount(self, moles: list):
        self.chem_state.setSpeciesAmounts(moles)

    def solve_chemical_equilibrium(self):
        self.chem_results = self.chem_equi_solver.solve(self.chem_state)
        self.chem_prop = self.chem_state.properties()

    def _get_species_amounts(self):
        return self.chem_state.speciesAmounts()

    def _get_species_log_activity_coeffs(self):
        return self.chem_prop.lnActivityCoefficients().val

    def _get_fluid_density(self):
        """The unit of density is kg/m3."""
        return self.chem_prop.phaseDensities().val[0]
