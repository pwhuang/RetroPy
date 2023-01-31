# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *
import numpy as np

class TracerTransportProblemExp(TracerTransportProblem):
    """
    A class that solves single-phase tracer transport problems by solving the
    exponents of the tracer concentration. This formulation guarantees the non-
    negativity of such concentration.
    """

    def initialize_form(self):
        """"""

        self.__u = TrialFunction(self.comp_func_spaces)
        self._TracerTransportProblem__w = TestFunction(self.comp_func_spaces)
        self.__w = self._TracerTransportProblem__w

        self._TracerTransportProblem__u = as_vector([exp(u) for u in self.__u])

        u = self._TracerTransportProblem__u

        self.tracer_forms = [Constant(self.mesh, 0.0)*inner(self.__w, u)*self.dx]*super().num_forms

    def save_to_file(self, time, is_exponentiated=False, is_saving_pv=False):
        """"""

        try:
            self.outputter
        except:
            return False

        func_to_save = self.fluid_components

        if self.num_component==1:
            self.output_assigner.assign(self.output_func_list[0], func_to_save)
        else:
            self.output_assigner.assign(self.output_func_list, func_to_save)

        for key, i in self.component_dict.items():
            if is_exponentiated:
                pass
            else:
                self.output_func_list[i].vector[:] = \
                np.exp(self.output_func_list[i].vector[:])

            self.write_function(self.output_func_list[i], key, time)

        if is_saving_pv:
            self.save_fluid_pressure(time)
            self.save_fluid_velocity(time)

        return True

    def get_fluid_components(self):
        return as_vector([exp(self.fluid_components[i]) for i in range(self.num_component)])

    def set_component_ics(self, expressions):
        super().set_component_ics(expressions)

        if np.any(self.fluid_components.vector[:] < DOLFIN_EPS):
            raise ValueError('fluid_components contains negative or zero values!')

        self.logarithm_fluid_components()

    def logarithm_fluid_components(self):
        self.fluid_components.vector[:] = np.log(self.fluid_components.vector[:])
