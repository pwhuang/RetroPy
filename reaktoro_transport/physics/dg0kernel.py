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

        return dot(w, u-u0)/self.dt*self.dx

    def advection_flux_limited(self, w, u, u_up, kappa=-1.0, marker=0):
        """
        We implemented the kappa interpolation scheme originated from van Leer
        (1985).
        ---------------------------------
        kappa = -1 : Second order upwind
                 0 : Fromm's scheme
                1/2: QUICK
                1/3: Third order upwind
                 1 : Centered scheme
        """

        eps = Constant(1e-13)

        adv = self.fluid_velocity
        n = self.n

        adv_np = (dot(adv, n) + Abs(dot(adv, n))) / 2.0
        adv_nm = (dot(adv, n) - Abs(dot(adv, n))) / 2.0

        np = sign(adv_np)
        nm = sign(adv_nm)

        grad_down = jump(adv_nm*u) - jump(adv_np*u)
        grad_up = jump(adv_np*u) - jump(adv_np*u_up)

        down = jump(nm*u) - jump(np*u)
        up = jump(np*u) - jump(np*u_up)

        r = as_vector([down[i]/(up[i] + eps)\
                      for i in range(self.num_component)])

        high_order_flux = Constant((1.0-kappa)/4.0)*grad_up\
                        + Constant((1.0+kappa)/4.0)*grad_down

        advective_flux = as_vector([self.flux_limiter(r[i])*high_order_flux[i]\
                                for i in range(self.num_component)])

        return dot(jump(w), advective_flux)*self.dS(marker)

    def flux_limiter(self, r):
        return FluxLimiterCollection.minmod(r)

    def general_flux_bc(self, w, value, marker: int):
        return w*value*self.ds(marker)

    def diffusion_flux_bc(self, w, u, D, value, marker: int):
        """"""

        # Note, this is only a (not good enough) approximation of the flux.
        dh = sqrt(dot(self.boundary_cell_coord - self.cell_coord,
                      self.boundary_cell_coord - self.cell_coord))

        return self.general_flux_bc(w, D*(u - value)/dh, marker)

    def advection_flux_bc(self, w, value, marker: int):
        """"""

        adv = dot(self.fluid_velocity, self.n)
        return w*adv*value*self.ds(marker)

    def advection_outflow_bc(self, w, u, marker: int):
        """"""

        adv = dot(self.fluid_velocity, self.n)
        return inner(w, adv*u)*self.ds(marker)
