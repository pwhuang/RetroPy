# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *

class StokesFlowBase(FluidProperty):
    """The base class for Stokes flow problems."""

    def __init__(self, mesh, boundary_markers, domain_markers):
        self.set_mesh(mesh)
        self.set_boundary_markers(boundary_markers)
        self.set_domain_markers(domain_markers)

    def __init_function_space(self):
        func_space_list = [self.velocity_finite_element,
                           self.pressure_finite_element]

        self.mixed_func_space = FunctionSpace(self.mesh,
                                              MixedElement(func_space_list))

    def __init_function_assigner(self):
        self.velocity_assigner = FunctionAssigner(self.velocity_func_space,
                                                  self.mixed_func_space.sub(0))

        self.pressure_assigner = FunctionAssigner(self.pressure_func_space,
                                                  self.mixed_func_space.sub(1))

        self.mixed_v_assigner = FunctionAssigner(self.mixed_func_space.sub(0),
                                                 self.velocity_func_space)

        self.mixed_p_assigner = FunctionAssigner(self.mixed_func_space.sub(1),
                                                 self.pressure_func_space)

    def mark_flow_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Keywords
        --------
        inlet : Sets the boundary flow rate.
        noslip : Sets the boundary to no-slip boundary condition.
        velocity_bc : User defined velocity boundary condition.
        """

        self.stokes_boundary_dict = kwargs

    def set_pressure_ic(self, init_cond_pressure):
        """Sets up the initial condition of pressure."""
        self.init_cond_pressure = init_cond_pressure
        self.fluid_pressure.interpolate(init_cond_pressure)

    def set_pressure_bc(self, pressure_bc):
        self.pressure_bc = pressure_bc

        v = self.__v
        n = self.n
        ds = self.ds
        mu = self._mu
        u0 = self.__U1.sub(0)

        for i, marker in enumerate(self.stokes_boundary_dict['inlet']):
            self.residual_momentum_form += \
            self.pressure_bc[i]*inner(n, v)*ds(marker) \
            - mu*inner(dot(grad(u0), n), v)*ds(marker)

    def set_velocity_bc(self, velocity_bc_val: list):
        """
        Arguments
        ---------
        velocity_bc_val : list of Constants,
                          e.g., [Constant((1.0, -1.0)), Constant((0.0, -2.0))]
        """

        if self.mesh.geometric_dimension()==2:
            noslip = Constant((0.0, 0.0))
        elif self.mesh.geometric_dimension()==3:
            noslip = Constant((0.0, 0.0, 0.0))

        self.velocity_bc = []

        for marker in self.stokes_boundary_dict['noslip']:
            self.velocity_bc.append(DirichletBC(self.velocity_func_space,
                                                noslip,
                                                self.boundary_markers, marker))

        for i, marker in enumerate(self.stokes_boundary_dict['velocity_bc']):
            self.velocity_bc.append(DirichletBC(self.velocity_func_space,
                                                velocity_bc_val[i],
                                                self.boundary_markers, marker))

        self.mixed_vel_bc = []
        self.mixed_pres_bc = []

        for marker in self.stokes_boundary_dict['noslip']:
            self.mixed_vel_bc.append(DirichletBC(self.mixed_func_space.sub(0), noslip,
                                                 self.boundary_markers, marker))

        for i, marker in enumerate(self.stokes_boundary_dict['velocity_bc']):
            self.mixed_vel_bc.append(DirichletBC(self.mixed_func_space.sub(0),
                                                 velocity_bc_val[i],
                                                 self.boundary_markers, marker))

    def generate_residual_form(self):
        """"""
        self.__init_function_space()
        self.__init_function_assigner()

        W = self.mixed_func_space

        (self.__u, self.__p) = TrialFunctions(W)
        (self.__v, self.__q) = TestFunctions(W)

        self.__U0 = Function(W)
        self.__U1 = Function(W)

        #self.__U0.vector()[:] = 1.0

        # V = self.velocity_func_space
        # Q = self.pressure_func_space
        #
        # self.__v, self.__q = TestFunction(V), TestFunction(Q)

        #self.__v, self.__q = self.__U0.sub(0), self.__U0.sub(1)
        u, p = self.__u, self.__p
        v, q = self.__v, self.__q
        #u0, p0 = self.fluid_velocity, self.fluid_pressure
        u0, p0 = self.__U1.sub(0), self.__U1.sub(1)

        mu, rho, g = self._mu, self._rho, self._g

        n = self.n
        dx, ds = self.dx, self.ds

        self.residual_momentum_form = mu*inner(grad(v), grad(u))*dx - inner(div(v), p)*dx
        self.residual_momentum_form -= inner(v, rho*g)*dx
        self.residual_momentum_form += q*div(rho*u)*dx

        self.__A = PETScMatrix()
        self.__b = PETScVector()

    def add_mass_source_to_residual_form(self, sources: list):
        q = self.__q

        for source in sources:
            self.residual_mass_form -= q*source*self.dx

    def add_momentum_source_to_residual_form(self, sources: list):
        v  = self.__v

        for source in sources:
            self.residual_momentum_form -= inner(v, source)*self.dx

    def get_flow_residual(self):
        """"""

        self.__rmf, self.__L = lhs(self.residual_momentum_form), rhs(self.residual_momentum_form)
        #u0, p0 = self.fluid_velocity, self.fluid_pressure

        # self.velocity_assigner.assign(self.__U0.sub(0), self.fluid_velocity)
        # self.pressure_assigner.assign(self.__U0.sub(1), self.fluid_pressure)
        self.mixed_v_assigner.assign(self.__U1.sub(0), self.fluid_velocity)
        self.mixed_p_assigner.assign(self.__U1.sub(1), self.fluid_pressure)

        assemble_system(self.__rmf, self.__L, self.mixed_vel_bc, A_tensor=self.__A, b_tensor=self.__b)
        # assemble(self.__rmf, tensor=self.__A)
        # assemble(self.__L, tensor=self.__b)

        # for bc in self.mixed_vel_bc:
        #     #bc.apply(self.__A, self.__b)
        #     bc.apply(self.__U1.vector())

        #self.__U0.vector()[:] = 1.0
        # for bc in self.mixed_vel_bc:
        #     bc.apply(self.__U0.vector())

        #action(self.residual_momentum_form, self.__U1)

        #residual_momentum = assemble(self.residual_momentum_form)

        # for bc in self.mixed_vel_bc:
        #     bc.apply(residual_momentum, self.__U0.vector())
        #residual_mass = assemble(self.residual_mass_form)

        #residual = residual_momentum.norm('l2')
        #residual = abs(residual_momentum)
        #residual += residual_mass.norm('l2')
        #print((self.__A*self.__x - self.__b).get_local()[:])
        residual = (self.__A*self.__U1.vector() - self.__b).norm('l2')

        return residual
