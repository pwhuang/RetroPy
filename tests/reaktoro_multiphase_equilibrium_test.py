# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

from retropy.problem import MassBalanceBase
from retropy.manager import ReactionManager

import reaktoro as rkt
import numpy as np

class EquilibriumProblem(MassBalanceBase, ReactionManager):
    def initialize_Reaktoro(self):
        """
        """

        db = rkt.SupcrtDatabase("supcrt07")

        aqueous_components = self.component_str + ' ' + self.solvent_name

        self.aqueous_phase = rkt.AqueousPhase(aqueous_components)
        self.gaseous_phase = rkt.GaseousPhase('CO2(g)')
        self.chem_system = rkt.ChemicalSystem(db, *[self.aqueous_phase, self.gaseous_phase])

        self.set_activity_models()

        self.chem_equi_solver = rkt.EquilibriumSolver(self.chem_system)

        self.chem_state = rkt.ChemicalState(self.chem_system)
        self.chem_prop = rkt.ChemicalProps(self.chem_state)
        self.aqueous_prop = rkt.AqueousProps(self.chem_state)

        self.one_over_ln10 = 1.0/np.log(10.0)

    def set_activity_models(self):
        self.aqueous_phase.set(rkt.chain(rkt.ActivityModelHKF(),
                                         rkt.ActivityModelDrummond('CO2(aq)')))
        self.gaseous_phase.set(rkt.ActivityModelPengRobinson())

try:
    problem = EquilibriumProblem()
    problem.set_components('Na+ Cl- H+ OH- CO2(aq) CO3-2 HCO3-')
    problem.H_idx = problem.component_dict['H+']
    problem.set_solvent('H2O(aq)')
    problem.initialize_Reaktoro()

    problem._set_temperature(298.15, 'K')
    problem._set_pressure(101325, 'Pa')
    problem._set_species_amount([1.0, 1.0, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 55.0, 1000.0])

    problem.solve_chemical_equilibrium()

    print(problem._get_species_amounts())
    print(problem._get_fluid_density())
    print(problem._get_fluid_pH())

    test_result = True

except:
    print('Expection occurred!', sys.exc_info())
    test_result = False

def test_function():
    assert test_result
