from dolfin import *
parameters["ghost_mode"] = "shared_vertex"

from ufl.algebra import Abs
from ufl.operators import sqrt
from ufl import min_value, max_value, sign

from ..material import FluidProperty, ComponentProperty

from .transport_problem_base import TransportProblemBase
from .mass_balance_base import MassBalanceBase

from .tracer_transport_problem import TracerTransportProblem
from .stokesflow_uzawa import StokesFlowUzawa

from .darcyflow_base import DarcyFlowBase
from .darcyflow_uzawa import DarcyFlowUzawa
from .darcyflow_mixedpoisson import DarcyFlowMixedPoisson
from .darcyflow_angot import DarcyFlowAngot
