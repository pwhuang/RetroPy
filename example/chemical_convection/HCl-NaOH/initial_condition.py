# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.problem import MassBalanceBase
from retropy.manager import ReactionManager

class EquilibriumProblem(MassBalanceBase, ReactionManager):
    pass

init_cond = EquilibriumProblem()
init_cond.set_components('Na+', 'Cl-', 'H+', 'OH-')
init_cond.set_solvent('H2O(l)')

init_cond.initialize_Reaktoro()
init_cond._set_temperature(298.15, 'K')
init_cond._set_pressure(1e5, 'Pa')
init_cond._set_species_amount([1.0, 1e-15, 1e-15, 1.0, 55.36])

init_cond.solve_chemical_equilibrium()

print(f"The NaOH solution has the volume of {init_cond._get_fluid_volume()*1e3} Liters.")
print(f"The NaOH solution has the wt% of {(22.99 + 17.0)/(55.36*18.0 + 22.99 + 17.0)*1e2:.3f} %.")
print(f"The NaOH solution has the density of {init_cond._get_fluid_density()} kg/m3.")

init_cond._set_species_amount([1e-15, 1.0, 1.0, 1e-15, 54.17])
init_cond.solve_chemical_equilibrium()

print(f"The HCl solution has the volume of {init_cond._get_fluid_volume()*1e3} Liters.")
print(f"The HCl solution has the wt% of {(35.453 + 1.0)/(54.17*18.0 + 35.453 + 1.0)*1e2:.3f} %.")
print(f"The HCl solution has the density of {init_cond._get_fluid_density()} kg/m3.")
