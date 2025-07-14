# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedLineMesh
from retropy.problem import TracerTransportProblem

from dolfinx.fem import functionspace, Function, Constant, assemble_scalar, form
from ufl import as_vector

import numpy as np
import pytensor
from pytensor import tensor as pt
from scipy.optimize import fsolve
from mpi4py import MPI


class BuckleyLeverett(TracerTransportProblem):
    """
    This benchmark problem benchmarks 1D multiphase flow using the Buckley-Leverett problem.
    """

    
    @staticmethod
    def initial_expr():
        return lambda x: 0.0 * x[0] + 0.1

    def get_mesh_and_markers(self, nx):
        marked_mesh = MarkedLineMesh(xmin=0.0, xmax=1.0, num_elements=nx)
        self.mesh_characteristic_length = 1.0 / nx

        return marked_mesh

    def set_flow_field(self):
        self.fluid_velocity = Function(self.Vec_CG1_space)
        self.fluid_velocity.x.array[:] = 1.0
        self.fluid_velocity.x.scatter_forward()
        self.set_advection_velocity()

    def define_problem(self):
        self.set_components("Sw")
        self.set_component_fe_space()
        self.initialize_form()

        self.mark_component_boundary(
            {"Sw": [self.marker_dict["left"], self.marker_dict["right"]]}
        )

        self.set_component_ics("Sw", self.initial_expr())

    @staticmethod
    def fractional_flow(S, b):
        return S**2 / (S**2 + b * (1.0 - S)**2)

    def add_physics_to_form(self, u, **kwargs):
        one = Constant(self.mesh, 1.0)
        self.__b = Constant(self.mesh, 1.5)
        b = self.__b.value
        S_inlet = 1.0

        self.inlet_flux = Constant(self.mesh, -1.0 * self.fractional_flow(S_inlet, b))
        self.outlet_flux = self.fractional_flow(self.fluid_comp_sub[0], b)

        self.add_time_derivatives(u)
        self.add_explicit_advection(as_vector([self.fractional_flow(self.fluid_comp_sub[0], b)]), 
                                    kappa=one, marker=0)
        
        self.add_component_flux_bc("Sw", [self.inlet_flux, self.outlet_flux], kappa=one)
        self.add_outflow_bc(u)

    def get_solution(self, t_end):
        S = pt.scalar("S", dtype="float64")

        f = self.fractional_flow(S, self.__b.value)

        df = pytensor.grad(f, S)
        dfdS = pytensor.function([S], df)
        f_func = pytensor.function([S], f)

        # Welge's method for determining saturation at the front
        S_i = pt.scalar("S_i", dtype="float64")

        W = S - S_i + (self.fractional_flow(S_i, self.__b.value) - f ) / df
        W_func_np = np.vectorize(pytensor.function([S, S_i], W))

        # Calculate shock speed
        S_i = self.s_init
        S_r = fsolve(W_func_np, 0.99, args=(self.s_init))[0]
        shock_speed = (f_func(S_r) - f_func(S_i)) / (S_r - S_i)

        # Calculate saturation as a function of space and time
        x = pt.scalar("x", dtype="float64")
        t = pt.scalar("t", dtype="float64")

        func_np = np.vectorize(pytensor.function([S, x, t], df - x / t))

        x_space = self.vertex_coord.x.array
        x_mid = 0.5 * (x_space[:-1] + x_space[1:])
        
        solution = np.ones_like(x_mid) * S_i
        rarefaction_cell_count = np.sum(x_space < shock_speed * t_end)
        idx = rarefaction_cell_count - 1

        for i in range(idx):
            solution[i] = fsolve(func_np, 1.0, args=(x_mid[i], t_end))[0]
            
        S_intersect = fsolve(func_np, 1.0, args=(shock_speed, 1.0))[0]
        solution[idx] = 0.5 * (fsolve(func_np, 1.0, args=(x_space[idx], t_end))[0] + S_intersect)

        self.solution = Function(self.comp_func_spaces)
        self.solution.sub(0).x.array[:] = solution

        return self.solution

    def get_error_norm(self):
        comm = self.mesh.comm
        mass_error = self.fluid_components - self.solution
        mass_error_norm = assemble_scalar(form(mass_error**2 * self.dx))

        return np.sqrt(comm.allreduce(mass_error_norm, op=MPI.SUM))
