# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx.fem import (Function, FunctionSpace, VectorFunctionSpace,
                         dirichletbc, locate_dofs_topological, Constant)
from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from ufl.algebra import Abs
from ufl import (lhs, rhs, action, derivative, TrialFunction)

from .reactive_transport_problem_base import reactive_transport_problem_base
from .multicomponent_diffusion_problem import multicomponent_diffusion_problem
from .multicomponent_transport_problem import multicomponent_transport_problem

def set_default_solver_parameters(prm):
    prm.setTolerances(rtol = 1e-12, atol=1e-14, divtol=None, max_it=5000)
    # prm['error_on_nonconvergence'] = True
    # prm['monitor_convergence'] = True
    # prm['nonzero_initial_guess'] = True

from .steady_state_solver import SteadyStateSolver
from .transient_solver import TransientSolver
from .transient_rk2_solver import TransientRK2Solver
from .projection_solver import ProjectionSolver
from .gradient_solver import GradientSolver
from .electric_field_solver import ElectricFieldSolver
from .transient_nl_solver import TransientNLSolver
from .custom_nl_solver import CustomNLSolver
