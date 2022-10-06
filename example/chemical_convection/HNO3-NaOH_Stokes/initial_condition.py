# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from reaktoro_transport.problem import MassBalanceBase
from reaktoro_transport.manager import ReactionManager

class EquilibriumProblem(MassBalanceBase, ReactionManager):
    pass

init_cond = EquilibriumProblem()
init_cond.set_components('Na+', 'NO3-', 'H+', 'OH-')
init_cond.set_solvent('H2O(l)')

init_cond.initialize_Reaktoro()
init_cond._set_temperature(298.15, 'K')
init_cond._set_pressure(1e5, 'Pa')
init_cond._set_species_amount([1.4, 1e-15, 1e-15, 1.4, 55.361])

init_cond.solve_chemical_equilibrium()

print(f"The NaOH solution has the volume of {init_cond._get_fluid_volume()*1e3} Liters.")
print(f"The NaOH solution has the wt% of {(22.99 + 17.0)*1.4/(55.361*18.0 + (22.99 + 17.0)*1.4)*1e2:.3f} %.")
print(f"The NaOH solution has the density of {init_cond._get_fluid_density()} kg/m3.")

init_cond._set_species_amount([1e-15, 1.5, 1.5, 1e-15, 52.712])
init_cond.solve_chemical_equilibrium()

print(f"The HNO3 solution has the volume of {init_cond._get_fluid_volume()*1e3} Liters.")
print(f"The HNO3 solution has the wt% of {(62.0049 + 1.0)*1.5/(52.712*18.0 + (62.0049 + 1.0)*1.5)*1e2:.3f} %.")
print(f"The HNO3 solution has the density of {init_cond._get_fluid_density()} kg/m3.")
