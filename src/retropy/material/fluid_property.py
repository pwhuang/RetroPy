# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx.fem import Function, Constant
from petsc4py.PETSc import ScalarType

class FluidProperty:
    """This class defines the physical properties of fluids.
    Used as components for problem classes.
    """

    def set_permeability(self, permeability):
        """Sets the permeability in the unit of length squared."""

        self._k = Function(self.DG0_space)

        if type(permeability)==float:
            self._k.x.array[:] = permeability
            self._k.x.scatter_forward()
        else:
            self._k.interpolate(permeability)

    def set_porosity(self, porosity):
        """Sets the porosity in dimensionless unit."""
        
        self._phi = Function(self.DG0_space)

        if type(porosity)==float:
            self._phi.x.array[:] = porosity
            self._phi.x.scatter_forward()
        else:
            self._phi.interpolate(porosity)

    def set_fluid_density(self, density):
        """Sets the fluid density in the unit of mass over volume."""

        self._rho = Function(self.DG0_space)

        if type(density)==float:
            self._rho.x.array[:] = density
            self._rho.x.scatter_forward()
        else:
            self._rho.interpolate(density)

        self.fluid_density = self._rho
        self.fluid_density.name = 'fluid_density'

    def set_fluid_viscosity(self, viscosity: float):
        """Sets fluid dynamic viscosity in the unit of pressure*time."""

        self._mu = Constant(self.mesh, ScalarType(viscosity))

    def set_gravity(self, gravity: tuple):
        """Sets up the gravity in the body force term of Darcy's law."""

        self._g = Constant(self.mesh, ScalarType(gravity))

    def get_permeability(self):
        return self._k
