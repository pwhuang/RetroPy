import sys
sys.path.insert(0, '../..')

import numpy as np
import reaktoro_transport.solver as solver

import reaktoro as rkt
from dolfin import *


# Comparing the solution of the concentration transport in the Cartesian coordinates
# with respect to the transient asymptotic solution
# The asymtotic solution is derived by assuming Da is in the order of epsilon squared,
# with left boundary equals to 0 and C(x, t=0) = 0.

def adv_diff_reac_transient_sol_fracture(Pe, Da, eps, C_b, x_space, t):
    alpha2 = 2*Da/eps**2

    n_space = np.arange(2,21,1)

    C = np.zeros_like(x_space)

    # We still need to find out how to better calculate G_n

    n = 1
    G_n = np.log(2)
    J_n = 12*alpha2 + (2*n-1)**2*(Pe*G_n + 3*np.pi**2)

    C += (1-np.exp(-J_n*t/12))/J_n\
    *np.sin((2*n-1)/2*np.pi*x_space)/(2*n-1)**2

    for n in n_space:
        d = int(n/2)
        G_n = (np.math.factorial(d)*2**(d+1)*3**(d+0.5)/np.pi**(d+1)/np.prod(np.arange(1,2*d+1,2)) + np.log(2))/(2*n-1)
        J_n = 12*alpha2 + (2*n-1)**2*(Pe*G_n + 3*np.pi**2)

        C += (1-np.exp(-J_n*t/12))/J_n\
        *np.sin((2*n-1)/2*np.pi*x_space)/(2*n-1)

    C = 48*alpha2/np.pi*C

    return C

def reaktoro_init():
    db = rkt.Database('supcrt98.xml')

    # Step 3: Define the chemical system
    editor = rkt.ChemicalEditor(db)
    #editor.addAqueousPhaseWithElements('H O Na Cl')
    editor.addAqueousPhase('H2O(l) NaCl(aq) H+ OH-')
    editor.addMineralPhase('Halite')

    MinReact = editor.addMineralReaction('Halite')

    MinReact.setEquation('Halite = NaCl(aq)')
    MinReact.addMechanism('logk = -0.25 mol/(m2*s); Ea = 7.4 kJ/mol')
    #MinReact.setSpecificSurfaceArea(1, 'm2/kg')

    MinReact.setSurfaceArea(1, 'm2')

    # Step 4: Construct the chemical system
    system = rkt.ChemicalSystem(editor)
    reactions = rkt.ReactionSystem(editor)

    partition = rkt.Partition(system)
    partition.setKineticSpecies(['Halite'])

    path = rkt.KineticPath(reactions)
    path.setPartition(partition)

    # Step 5: Define the chemical equilibrium problem
    problem = rkt.EquilibriumProblem(system)
    problem.setTemperature(25, 'celsius')
    problem.setPressure(1, 'bar')

    problem.setElementAmount('H', 110689.1155)
    problem.setElementAmount('O', 110689.1155*2)

    # Step 6: Calculate the chemical equilibrium state
    state0 = rkt.equilibrate(problem)
    #print(state0)
    #state0.scalePhaseVolume("Aqueous", 1, "m3")

    state0.setSpeciesMass('Halite', 1000, 'kg')

    # Get the concentration scale by equilibrate the problem with Halite
    problem.add('Halite', 1000, 'kg')
    state_eq = rkt.equilibrate(problem)
    C_scale = state_eq.speciesAmount('NaCl(aq)')

    return state0, path, reactions, C_scale

nx_nd = 30
ny_nd = 30
mesh_2d = UnitSquareMesh(nx_nd, ny_nd)


Lx = 5e-3 #8e-3 #m
Ly = 5e-4

kp_over_Ly = 8.75e-5 # m/sec
kp = kp_over_Ly*Ly

Dc = 1e-9 #Diffusivity m^2/sec
uc = 1e-6 #Characteristic velocity m/sec

epsilon = Ly/Lx
Pe = Lx*uc/Dc
Da = Ly*kp/Dc
alpha = Da/Pe/epsilon**2
t_scale = Lx**2/Dc

C_b = 0  # Concentration boundary condition at the left boundary
initial_expr = Expression('0', degree=1)
dt_num = 5e-3
time_steps = 20
theta_num = 0.5 # The Crank-Nicolson Scheme

C = solver.concentration_transport2D_reaktoro(mesh_2d, epsilon, Pe, C_b, initial_expr, t_scale\
                                              , dt_num, time_steps, theta_num, reaktoro_init)

# Picking the last time step to compare
C_vertex = C[-1].compute_vertex_values(mesh_2d).reshape(nx_nd+1, ny_nd+1)

x_space = np.linspace(0,1,nx_nd+1)
asym_sol = adv_diff_reac_transient_sol_fracture(Pe, Da, epsilon, C_b, x_space, dt_num*time_steps)

print(np.linalg.norm(asym_sol-C_vertex[0,:]))

tolerance = 0.1
def test_answer():
    assert np.linalg.norm(asym_sol-C_vertex[0,:]) < tolerance
