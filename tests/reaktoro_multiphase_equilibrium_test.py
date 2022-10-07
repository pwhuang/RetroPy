# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

from retropy.problem import MassBalanceBase
from retropy.manager import ReactionManager

class EquilibriumProblem(MassBalanceBase, ReactionManager):
    def set_chem_editor(self, database):
        editor = super().set_chem_editor(database)
        self.gaseous_phase = editor.addGaseousPhase(['CO2(g)'])

        return editor

    def set_activity_models(self):
        self.aqueous_phase.setChemicalModelHKF()
        self.aqueous_phase.setActivityModelDrummondCO2()
        self.gaseous_phase.setChemicalModelPengRobinson()

try:
    problem = EquilibriumProblem()
    problem.set_components('Na+', 'Cl-', 'H+', 'OH-', 'CO2(aq)', 'CO3--', 'HCO3-')
    problem.H_idx = problem.component_dict['H+']
    problem.set_solvent('H2O(l)')
    problem.initialize_Reaktoro()

    problem._set_temperature(298.15, 'K')
    problem._set_pressure(101325, 'Pa')
    problem._set_species_amount([1.0, 1.0, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 55.0, 1000.0])

    problem.solve_chemical_equilibrium()

    print(problem._get_species_amounts())
    print(problem._get_species_log_activity_coeffs())
    print(problem._get_species_chemical_potentials())
    print(problem._get_fluid_density())
    print(problem._get_fluid_pH())

    test_result = True

except:
    print('Expection occurred!', sys.exc_info())
    test_result = False

def test_function():
    assert test_result
