# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx import *
# parameters["ghost_mode"] = "shared_vertex"
# parameters["form_compiler"]["optimize"] = True
# parameters["form_compiler"]["cpp_optimize"] = True
# parameters["std_out_all_processes"] = False

from ufl.algebra import Abs
from ufl.operators import sqrt
from ufl import min_value, max_value, sign

from ..material import FluidProperty, ComponentProperty

from .transport_problem_base import TransportProblemBase
from .mass_balance_base import MassBalanceBase

from .tracer_transport_problem import TracerTransportProblem
from .tracer_transport_problem_exp import TracerTransportProblemExp

from .stokesflow_base import StokesFlowBase
from .stokesflow_uzawa import StokesFlowUzawa
from .stokesflow_mixedpoisson import StokesFlowMixedPoisson

from .darcyflow_base import DarcyFlowBase
from .darcyflow_uzawa import DarcyFlowUzawa
from .darcyflow_mixedpoisson import DarcyFlowMixedPoisson
from .darcyflow_angot import DarcyFlowAngot
