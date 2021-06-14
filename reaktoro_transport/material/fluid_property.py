from dolfin import Expression, interpolate, Constant

class FluidProperty:
    """This class defines the physical properties of fluids.
    Used as components for problems.
    """

    def set_permeability(self, permeability: Expression):
        """Sets the permeability in the unit of length squared."""

        self._k = interpolate(permeability, self.pressure_func_space)

    def set_porosity(self, porosity: Expression):
        """Sets the porosity in dimensionless unit."""

        self._phi = interpolate(porosity, self.pressure_func_space)

    def set_fluid_density(self, density: float):
        """Sets the fluid density in the unit of mass over volume."""

        self._rho = Constant(density)

    def set_fluid_viscosity(self, viscosity: float):
        """Sets fluid dynamic viscosity in the unit of pressure*time."""

        self._mu = Constant(viscosity)

    def set_gravity(self, gravity: tuple):
        """Sets up the gravity in the body force term of Darcy's law."""

        self._g = Constant(gravity)

    def get_permeability(self):
        return self._k
