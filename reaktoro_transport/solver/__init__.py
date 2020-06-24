#All the necessities in your life
import numpy as np
import reaktoro as rkt
from dolfin import *
import dolfin
from dolfin import inner, grad, Constant

import sys
sys.path.insert(0, '../..')

import reaktoro_transport.tools as tools

#Importing things from the same directory
from .stokes_lubrication import stokes_lubrication
from .stokes_lubrication_cylindrical import stokes_lubrication_cylindrical
from .stokes_lubrication_phase_field import stokes_lubrication_phase_field
from .stokes_uzawa import stokes_uzawa
from .transient_adv_diff_DG import transient_adv_diff_DG
from .concentration_transport2D import concentration_transport2D
from .concentration_transport_phase_field import concentration_transport_phase_field
from .concentration_transport2D_transient import concentration_transport2D_transient
from .concentration_transport2D_reaktoro import concentration_transport2D_reaktoro
from .reactive_transport_problem_base import reactive_transport_problem_base
from .multicomponent_diffusion_problem import multicomponent_diffusion_problem
