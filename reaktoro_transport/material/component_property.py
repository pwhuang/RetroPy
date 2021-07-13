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
        self._D = molecular_diffusivity

    def set_molar_mass(self, molar_mass: list[float]):
        pass

    def set_charge(self, charge: list[float]):
        pass
