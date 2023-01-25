# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace

# parameters["std_out_all_processes"] = False

from ufl.algebra import Abs
from ufl.operators import sqrt
from ufl import (min_value, max_value, sign, FacetNormal,
                 FacetArea, CellVolume, VectorElement, FiniteElement)

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
