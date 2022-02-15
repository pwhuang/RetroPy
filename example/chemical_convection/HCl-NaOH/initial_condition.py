from reaktoro_transport.problem import MassBalanceBase
from reaktoro_transport.manager import ReactionManager

class EquilibriumProblem(MassBalanceBase, ReactionManager):
    pass

init_cond = EquilibriumProblem()
init_cond.set_components('Na+', 'Cl-', 'H+', 'OH-')
init_cond.set_solvent('H2O(l)')

init_cond.initialize_Reaktoro()
init_cond._set_temperature(298.15, 'K')
init_cond._set_pressure(1.0, 'atm')
init_cond._set_species_amount([1.0, 1e-13, 1e-13, 1.0, 55.36])

init_cond.solve_chemical_equilibrium()

print(f"The NaOH solution has the volume of {init_cond._get_fluid_volume()*1e3} Liters.")
print(init_cond._get_fluid_density())

init_cond._set_species_amount([1e-13, 1.0, 1.0, 1e-13, 54.17])
init_cond.solve_chemical_equilibrium()

print(f"The HCl solution has the volume of {init_cond._get_fluid_volume()*1e3} Liters.")
print(init_cond._get_fluid_density())
