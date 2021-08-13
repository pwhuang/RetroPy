class ComponentProperty:
    """This class defines the physical properties of fluid components.
    Used as components for problem classes.
    """

    def set_molecular_diffusivity(self, molecular_diffusivity: list[float]):
        """
        Sets the molecular diffusivity in the unit of length squared over time.
        """

        if len(molecular_diffusivity)!=self.num_component:
            raise Exception("length of list != num_components")

        self.molecular_diffusivity = molecular_diffusivity
        self._D = self.molecular_diffusivity

    def set_molar_mass(self, molar_mass: list[float]):
        if len(molar_mass)!=self.num_component:
            raise Exception("length of list != num_components")

        self.molar_mass = molar_mass
        self._M = self.molar_mass

    def set_charge(self, charge: list[float]):
        if len(charge)!=self.num_component:
            raise Exception("length of list != num_components")

        self.charge = charge
        self._Z = self.charge
