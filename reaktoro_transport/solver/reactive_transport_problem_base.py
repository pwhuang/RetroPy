#from . import np, rkt
from . import *

class reactive_transport_problem_base:
    # Suitable for single phase reactive transport problems

    def __init__(self):
        self.bc_list = [] # When initialized, assume no boundary conditions.

    def set_chemical_system(self, component_list, pressure, temperature, molar_mass, diffusivity,\
                            charge, database='supcrt07.xml'):
        # Setup Reaktoro EquilibriumSolver

        self.component_list = component_list
        self.num_components = len(component_list)
        self.T = temperature
        self.M_list = molar_mass
        self.D_list = diffusivity
        self.z_list = charge

        db = rkt.Database(database)

        editor = rkt.ChemicalEditor(db)
        editor.addAqueousPhase(self.component_list)

        system = rkt.ChemicalSystem(editor)

        self.chem_equi_solver = rkt.EquilibriumSolver(system)
        self.chem_state = rkt.ChemicalState(system)
        self.chem_state.setPressure(pressure, 'atm')
        self.chem_state.setTemperature(temperature, 'K')
        self.chem_quant = rkt.ChemicalQuantity(self.chem_state)

    def set_mesh(self, mesh, boundary_markers):
        # Setup FeNiCs mesh and boundary markers
        self.mesh = mesh
        self.boundary_markers = boundary_markers
