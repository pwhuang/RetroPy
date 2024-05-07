# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedLineMesh
from retropy.problem import TracerTransportProblem

from dolfinx.fem import Function, Constant, assemble_scalar, form
# from ufl import exp, conditional, lt, as_vector
from mpi4py import MPI
import numpy as np


class ParticleAttachment(TracerTransportProblem):
    """
    This benchmark problem of particle transport and attachment is based on 
    the following work: Analytic solutions for colloid transport with time-
    and depth-dependent retention in porous media by Leij et. al., 2016, 
    published in Journal of Contaminant Hydrology, 195
    doi: 10.1016/j.jconhyd.2016.10.006
    """
    
    @staticmethod
    def irreversible_attachment(x, t, t0, Da, M):
        H = lambda x: np.heaviside(x, 0.5)
        G = np.exp(Da * x) - H(t - x) + (H(x - t + t0) - H(x - t)) * np.exp(Da / M * (t-x)) + H(t - t0 - x) * np.exp(Da / M * t0)

        G[G==np.nan] = 1.0

        C = (H(x - t + t0) - H(x - t)) * np.exp(-Da / M * (x - t)) / G
        S = 1. - np.exp(Da * x) / G

        return C, S

    def get_mesh_and_markers(self, nx):
        marked_mesh = MarkedLineMesh(xmin=0.0, xmax=1.0, num_elements=nx)
        self.mesh_characteristic_length = 1.0 / nx

        return marked_mesh

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        self.fluid_velocity = Function(self.Vec_CG1_space)
        self.fluid_velocity.interpolate(lambda x: (1.0 + 0.0 * x[0]))
        self.set_advection_velocity()

    def define_problem(self, Pe, Da_att, Da_det, M, t0):
        self.set_components("C S")
        self.set_component_fe_space()
        self.set_flow_field()
        self.initialize_form()

        Pe_inverse = Constant(self.mesh, 1.0 / Pe)
        self.Da_att = Constant(self.mesh, Da_att)
        self.Da_det = Constant(self.mesh, Da_det)
        self.__M = Constant(self.mesh, M)
        self.t0 = Constant(self.mesh, t0)
        
        zero = Constant(self.mesh, 0.0)

        self.set_molecular_diffusivity([Pe_inverse, zero])

        self.set_component_ics("C", lambda x: 0.0 * x[0])
        self.set_component_ics("S", lambda x: 0.0 * x[0])
        self.set_component_mobility([True, False])

        self.mark_component_boundary(
            {
                "C": [self.marker_dict["left"]],
                "outlet": [self.marker_dict["right"]],
            }
        )

    def langmuir_kinetics(self, C, S):
        one = Constant(self.mesh, 1.0)
        return self.Da_att * (one - S) * C - self.Da_det * S

    def add_physics_to_form(self, u, kappa=1.0, f_id=0):
        self.add_explicit_advection(u, kappa, marker=0, f_id=f_id)
        
        S, C = self.get_trial_function()[1], u[0]  # implicit in S, explicit in C

        self.add_mass_source(['C'], [-self.langmuir_kinetics(C, S)], kappa, f_id)
        self.add_mass_source(['S'], [self.langmuir_kinetics(C, S) / self.__M], kappa, f_id)

        self.inlet_flux = Constant(self.mesh, -1.0)
        self.add_component_flux_bc("C", [self.inlet_flux])
        self.add_outflow_bc(f_id)
        
    def generate_solution(self):
        x_space = self.cell_coord.x.array
        t, t0, Da_att, M = self.current_time.value, self.t0.value, self.Da_att.value, self.__M.value
        self.solution = Function(self.comp_func_spaces)        
        solution = np.zeros_like(self.solution.x.array)

        solution[::2], solution[1::2] = self.irreversible_attachment(x_space, t, t0, Da_att, M)

        self.solution.x.array[:] = solution
        self.solution.x.scatter_forward()

    def get_solution(self):
        return self.solution

    def get_error_norm(self):
        """
        This benchmark problem only compares the retention profile, 
        since the concentration profile is too diffusive to compare.
        """
        
        comm = self.mesh.comm
        mass_error = self.fluid_components.sub(1) - self.solution.sub(1)
        mass_error_norm = assemble_scalar(form(mass_error**2 * self.dx))

        return comm.allreduce(mass_error_norm, op=MPI.SUM) ** 0.5
