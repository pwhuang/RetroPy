from reaktoro_transport.problem import MassBalanceBase
from reaktoro_transport.manager import ReactionManager

class EquilibriumProblem(MassBalanceBase, ReactionManager):
    pass

init_cond = EquilibriumProblem()
init_cond.set_components('Li+', 'H+', 'OH-')
init_cond.set_solvent('H2O(l)')

init_cond.initialize_Reaktoro()
init_cond._set_temperature(298.15, 'K')
init_cond._set_pressure(1.0, 'atm')
init_cond._set_species_amount([0.01, 1e-15, 0.01, 55.345])

init_cond.solve_chemical_equilibrium()

print(f"The LiOH solution (0.01M) has the volume of {init_cond._get_fluid_volume()*1e3} Liters.")

init_cond._set_species_amount([0.1, 1e-15, 0.1, 55.343])
init_cond.solve_chemical_equilibrium()

print(f"The LiOH solution (0.1M) has the volume of  {init_cond._get_fluid_volume()*1e3} Liters.")
