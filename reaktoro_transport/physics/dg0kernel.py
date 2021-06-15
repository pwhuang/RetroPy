from . import *

class DG0Kernel:
    """This class collects physics of discontinuous Galerkin methods.

    Notation
    ---------
    w : dolfin TestFunction
    u : dolfin TrialFunction or Function
    """

    fe_space = 'DG'
    fe_degree = 0

    def diffusion(self, w, u, D, marker: int):
        """TPFA diffusion operator

        Arguments
        ---------
        D : dolfin Constant or Function
            The thermal or mass diffusivity for Fickian diffusion.
        delta_h : dolfin Function
            The length between two cells.
        """

        return D*dot(jump(w), jump(u))/self.delta_h*self.dS(marker)

    def advection(self, w, u, marker: int):
        """Upwind advection operator

        Arguments
        ---------
        fluid_velocity : dolfin Function
            The advection velocity.
        n : dolfin FacetNormal
        """

        adv_np = (dot(self.fluid_velocity, self.n)\
                  + Abs(dot(self.fluid_velocity, self.n)))/2.0

        return dot(jump(w), jump(adv_np*u))*self.dS(marker)

    def d_dt(self, w, u, u0):
        """time derivative operator"""

        return dot(w, (u-u0))/self.dt

    def advection_flux_limiter():
        """"""

        return # TODO: Add this method.

    def diffusion_flux_bc(self, w, value, marker: int):
        """"""

        return

    def advection_flux_bc(self, w, value, marker: int):
        """"""

        adv = dot(self.fluid_velocity, self.n)
        return w*adv*value*self.ds(marker)

    def advection_outflow_bc(self, w, u, marker: int):
        """"""

        adv = dot(self.fluid_velocity, self.n)
        return inner(w, adv*u)*self.ds(marker)
