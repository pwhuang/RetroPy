# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np

class ComponentProperty:
    """This class defines the physical properties of fluid components.
    Used as components for problem classes.
    """

    def set_component_mobility(self, idx: list):
        """
        Sets up the mobility of components.

        Input example: [True, True, False],
        where True stands for mobile and False stands for immobile.
        """

        self.component_mobility = np.array(idx)
        self.component_mobility_idx = np.argwhere(self.component_mobility==True)[0]

    def set_molecular_diffusivity(self, molecular_diffusivity):
        """
        Sets the molecular diffusivity in the unit of length squared over time.
        """

        self.molecular_diffusivity = molecular_diffusivity
        self._D = np.array(molecular_diffusivity)

    def set_molar_mass(self, molar_mass):
        if len(molar_mass)!=self.num_component:
            raise Exception("length of list != num_components")

        self.molar_mass = molar_mass
        self._M = np.array(molar_mass)

    def set_charge(self, charge):
        if len(charge)!=self.num_component:
            raise Exception("length of list != num_components")

        self.charge = charge
        self._Z = np.array(charge)
