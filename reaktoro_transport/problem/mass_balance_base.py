from . import *
import reaktoro as rkt
from numpy import array

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
        self._M_fraction = self._M/self.M_solvent

    def initiaize_ln_activity(self):
        self.ln_activity = Function(self.comp_func_spaces)
        self.ln_activity_dict = {}

        for comp_name, idx in self.component_dict.items():
            self.ln_activity_dict['lna_'+comp_name] = idx

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

        system = rkt.ChemicalSystem(editor)
        self.num_chem_elements = system.numElements()
        self.__zeros = array([0.0]*self.num_chem_elements)

        self.chem_problem = rkt.EquilibriumProblem(system)

        self.chem_equi_solver = rkt.EquilibriumSolver(system)

        self.chem_state = rkt.ChemicalState(system)
        self.chem_quant = rkt.ChemicalQuantity(self.chem_state)
        self.chem_prop = rkt.ChemicalProperties(system)

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
        self.chem_problem.setElectricalCharge(0.0)

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
