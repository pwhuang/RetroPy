from dolfin import Expression, interpolate, Constant

class FluidProperty:
    """This class defines the physical properties of fluids.
    Used as components for problem classes.
    """

    def set_permeability(self, permeability: Expression):
        """Sets the permeability in the unit of length squared."""

        if type(permeability)==float:
            permeability = Constant(permeability)

        self._k = interpolate(permeability, self.DG0_space)

    def set_porosity(self, porosity: Expression):
        """Sets the porosity in dimensionless unit."""

        if type(porosity)==float:
            porosity = Constant(porosity)

        self._phi = interpolate(porosity, self.DG0_space)

    def set_fluid_density(self, density: Expression):
        """Sets the fluid density in the unit of mass over volume."""

        if type(density)==float:
            density = Constant(density)

        self._rho = interpolate(density, self.DG0_space)
        self.fluid_density = self._rho
        self.fluid_density.rename('density', 'fluid_density')

    def set_fluid_viscosity(self, viscosity: float):
        """Sets fluid dynamic viscosity in the unit of pressure*time."""

        self._mu = Constant(viscosity)

    def set_gravity(self, gravity: tuple):
        """Sets up the gravity in the body force term of Darcy's law."""

        self._g = Constant(gravity)

    def get_permeability(self):
        return self._k
