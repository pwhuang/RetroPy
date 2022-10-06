# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *

class CGKernel:
    """This class collects physics of discontinuous Galerkin methods.

    Notation
    ---------
    w : dolfin TestFunction
    u : dolfin TrialFunction or Function
    """

    fe_space = 'CG'
    fe_degree = 1

    def diffusion(self, w, u, D, marker: int):
        """Diffusion operator

        Arguments
        ---------
        D : dolfin Constant or Function
            The thermal or mass diffusivity for Fickian diffusion.
        """

        return inner(grad(w), D*grad(u))*self.dx(marker)

    def advection(self, w, u, marker: int):
        """Advection operator"""

        adv = self.fluid_velocity
        return w*inner(adv, grad(u))*self.dx(marker)

    def d_dt(self, w, u, u0):
        """time derivative operator"""

        return dot(w, (u-u0))/self.dt
