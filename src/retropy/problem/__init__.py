# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx import default_real_type
from dolfinx.fem import (
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    locate_dofs_geometrical,
    form,
    Constant,
    petsc,
    assemble_scalar,
)
from dolfinx.fem.petsc import (
    LinearProblem,
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)
from dolfinx.mesh import exterior_facet_indices
from dolfinx.io.utils import XDMFFile

DOLFIN_EPS = 1e-16

from ufl.algebra import Abs
from ufl.operators import sqrt
from ufl import (
    min_value,
    max_value,
    sign,
    lhs,
    rhs,
    Measure,
    FacetNormal,
    FacetArea,
    CellVolume,
    Circumradius,
    CellDiameter,
    TestFunction,
    TestFunctions,
    TrialFunction,
    TrialFunctions,
    dot,
    inner,
    div,
    elem_div,
    elem_mult,
    avg,
    jump,
    as_vector,
    exp,
    grad,
    dx,
    ds,
    dS,
)
from basix.ufl import element, mixed_element

from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
from mpi4py import MPI
from typing import Any

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
