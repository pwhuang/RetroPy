from . import *

class MassBalanceBase:
    """Base class for mass balance problems"""

    def set_components(self, *args):
        """Sets up the component dictionary.

        Input example: 'Na+', 'Cl-'
        """

        self.component_dict = {comp: idx for idx, comp in enumerate(args)}
        self.num_component = len(self.component_dict)
