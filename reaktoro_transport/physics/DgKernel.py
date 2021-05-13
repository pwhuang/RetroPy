class DgKernel:
    """This class collects physics of discontinuous Galerkin methods.

    Notation
    ---------
    w : dolfin TestFunction
    u : dolfin TrialFunction or Function
    """

    def dg_diffusion_TPFA(self, w, u, D):
        """
        D : dolfin Constant or Function
            The thermal or mass diffusivity for Fickian diffusion.
        delta_h : dolfin Function
            The length between two cells.
        """

        return D*dot(jump(w), jump(u))/self.delta_h

    def dg_advection_upwind(self, w, u):
        """
        fluid_velocity : dolfin Function
            The advection velocity.
        n : dolfin FacetNormal
        """

        adv_np = (dot(self.fluid_velocity, self.n)\
                  + Abs(dot(self.fluid_velocity, self.n)))/2.0
        return jump(w)*jump(adv_np*u)

    def dg_advection_flux_limiter():
        """"""
        return # TODO: Add this method.
