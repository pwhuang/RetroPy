from reaktoro_transport.problem import MassBalanceBase
from reaktoro_transport.manager import ReactionManager

class EquilibriumProblem(MassBalanceBase, ReactionManager):
    def set_activity_models(self):
        self.aqueous_phase.setChemicalModelPitzerHMW()

init_cond = EquilibriumProblem()
init_cond.set_components('Li+', 'H+', 'OH-')
init_cond.set_solvent('H2O(l)')

init_cond.initialize_Reaktoro()
init_cond._set_temperature(298.15, 'K')
init_cond._set_pressure(1e5, 'Pa')
init_cond._set_species_amount([0.01, 1e-15, 0.01, 55.345])

init_cond.solve_chemical_equilibrium()

print(f"The LiOH solution (0.01M) has the volume of {init_cond._get_fluid_volume()*1e3} Liters.")
print(f"The LiOH solution has the density of {init_cond._get_fluid_density()} kg/m3.")

init_cond._set_species_amount([0.1, 1e-15, 0.1, 55.343])
init_cond.solve_chemical_equilibrium()

print(f"The LiOH solution (0.1M) has the volume of  {init_cond._get_fluid_volume()*1e3} Liters.")

init_cond = EquilibriumProblem()
init_cond.set_components('Na+', 'Cl-', 'H+', 'OH-')
init_cond.set_solvent('H2O(l)')

init_cond.initialize_Reaktoro()
init_cond._set_temperature(298.15, 'K')
init_cond._set_pressure(1e5, 'Pa')
init_cond._set_species_amount([2.0, 2.0, 1e-15, 1e-15, 53.010])

init_cond.solve_chemical_equilibrium()

print(f"The NaCl solution (2.0M) has the volume of  {init_cond._get_fluid_volume()*1e3} Liters.")
