from . import *

def get_mass_fraction(component_list, pressure, temperature, molar_mass, species_amount):
    # This function calculates the equilibrium state of species in units of moles,
    # and returns the mass fraction.
    # component_list = ['Na+', 'Cl-', 'H+', 'OH-', 'H2O(l)']
    # molar_mass  = [22.99e-3, 35.453e-3, 1.0e-3, 17.0e-3, 18.0e-3]
    # pressure = 1.0 #atm
    # temperature = 273.15+25 #K
    # species_amount: List of moles

    db = rkt.Database('supcrt07.xml')

    editor = rkt.ChemicalEditor(db)
    editor.addAqueousPhase(component_list)

    system = rkt.ChemicalSystem(editor)

    chem_equi_solver = rkt.EquilibriumSolver(system)
    chem_state = rkt.ChemicalState(system)
    chem_state.setPressure(pressure, 'atm')
    chem_state.setTemperature(temperature, 'K')

    chem_state.setSpeciesAmounts(species_amount)
    chem_equi_solver.solve(chem_state)
    mol_frac = chem_state.speciesAmounts()/np.sum(chem_state.speciesAmounts())
    M_bar = np.sum(mol_frac*np.array(molar_mass))
    mass_frac = mol_frac*np.array(molar_mass)/M_bar

    return mass_frac
