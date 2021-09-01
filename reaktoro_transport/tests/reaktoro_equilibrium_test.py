import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '../../')

from reaktoro_transport.problem import MassBalanceBase

try:
    problem = MassBalanceBase()
    problem.set_components('Na+', 'Cl-', 'H+', 'OH-')
    problem.set_solvent('H2O(l)')
    problem.initialize_Reaktoro()

    problem._set_temperature(298, 'K')
    problem._set_pressure(101325, 'Pa')
    problem._set_species_amount([1.0, 1.0, 1e-15, 1e-15, 55.0])

    problem.solve_chemical_equilibrium()

    print(problem._get_species_amounts())
    print(problem._get_species_log_activity_coeffs())
    print(problem._get_fluid_density())

    test_result = True

except:
    print('Expection occurred!', sys.exc_info())
    test_result = False

def test_function():
    assert test_result
