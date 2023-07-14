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

        super().initialize_form()
        self._TracerTransportProblem__u = as_vector([exp(u) for u in self._TracerTransportProblem__u])

    def save_to_file(self, time, is_exponentiated=True, is_saving_pv=False):
        """"""

        if is_exponentiated:
            pass
        else:
            self.fluid_components.x.array[:] = np.exp(self.fluid_components.x.array[:])

        return super().save_to_file(time, is_saving_pv=is_saving_pv)

    def set_component_ics(self, name, expressions):
        super().set_component_ics(name, expressions)

        if np.any(self.fluid_components.x.array[:] < DOLFIN_EPS):
            raise ValueError('fluid_components contain negative or zero values!')

    def logarithm_fluid_components(self):
        self.fluid_components.x.array[:] = np.log(self.fluid_components.x.array[:])
