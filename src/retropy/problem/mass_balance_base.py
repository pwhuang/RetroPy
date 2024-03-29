# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *

class MassBalanceBase:
    """Base class for mass balance problems"""

    def set_components(self, comp: str):
        """
        Sets up the component dictionary.

        Input example: 'Na+ Cl-'
        """

        self.component_str = comp
        self.component_dict = {comp: idx for idx, comp in enumerate(comp.split(' '))}
        self.num_component = len(self.component_dict)

    def set_solvent(self, solvent='H2O(l)'):
        """The solvent is not included in transport calculations."""

        self.solvent_name = solvent

    def set_solvent_molar_mass(self, solvent_molar_mass=18.0e-3):
        self.M_solvent = solvent_molar_mass

    def set_solvent_ic(self, init_expr):
        self.solvent = Function(self.DG0_space)
        self.solvent.interpolate(init_expr)
        self.solvent.name = self.solvent_name
        self._M_fraction = self._M/self.M_solvent

    def initiaize_ln_activity(self):
        self.ln_activity = Function(self.comp_func_spaces)
        self.ln_activity_dict = {}

        for comp_name, idx in self.component_dict.items():
            self.ln_activity_dict[f'lna_{comp_name}'] = idx

    def initialize_fluid_pH(self):
        try:
            self.component_dict['H+']
        except:
            raise Exception('H+ does not exist in the chemcial system.')

        self.fluid_pH = Function(self.DG0_space)
        self.fluid_pH.name = 'pH'
