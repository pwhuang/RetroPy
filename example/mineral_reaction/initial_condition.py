# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.problem import MassBalanceBase
from retropy.manager import ReactionManager

import reaktoro as rkt

class EquilibriumProblem(MassBalanceBase, ReactionManager):
    def set_chem_system(self, database):
        db = rkt.PhreeqcDatabase('pitzer.dat')
        aqueous_components = self.component_str + ' ' + self.solvent_name

        self.aqueous_phase = rkt.AqueousPhase(aqueous_components)
        self.mineral_phase = rkt.MineralPhase(self.mineral_name)
        self.chem_system = rkt.ChemicalSystem(db, self.aqueous_phase, self.mineral_phase)

    def set_activity_models(self):
        self.aqueous_phase.set(rkt.ActivityModelPitzer())

init_cond = EquilibriumProblem()
init_cond.set_components('H+ OH- Ca+2 HCO3- CO3-2 CO2')
init_cond.set_solvent('H2O')
init_cond.set_mineral("Calcite")

init_cond.initialize_Reaktoro()
init_cond._set_temperature(298.15, 'K')
init_cond._set_pressure(1e5, 'Pa')

C_co2 = 0.01   # mol/L
n_h2o = 55.336

init_cond._set_species_amount([1e-16, 1e-16, 1e-16, 1e-16, 1e-16, C_co2, n_h2o, 1e-16])
init_cond.solve_chemical_equilibrium()

print(f"The CO2 solution has the volume of {init_cond._get_fluid_volume()*1e3} Liters.")
print(f"The CO2 solution has the density of {init_cond._get_fluid_density()} kg/m3.")