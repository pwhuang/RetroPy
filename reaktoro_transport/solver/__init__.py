from dolfin import *
from ufl.algebra import Abs

from .reactive_transport_problem_base import reactive_transport_problem_base
from .multicomponent_diffusion_problem import multicomponent_diffusion_problem
from .multicomponent_transport_problem import multicomponent_transport_problem

def set_default_solver_parameters(prm):
    prm['absolute_tolerance'] = 1e-14
    prm['relative_tolerance'] = 1e-12
    prm['maximum_iterations'] = 5000
    prm['error_on_nonconvergence'] = True
    prm['monitor_convergence'] = True
    prm['nonzero_initial_guess'] = True

from .steady_state_solver import SteadyStateSolver
from .transient_solver import TransientSolver
from .transient_rk2_solver import TransientRK2Solver
from .projection_solver import ProjectionSolver
from .gradient_solver import GradientSolver
from .electric_field_solver import ElectricFieldSolver
from .transient_nl_solver import TransientNLSolver
