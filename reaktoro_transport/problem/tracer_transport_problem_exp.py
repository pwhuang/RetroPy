from . import *
from ufl import Index

class TracerTransportProblemExp(TracerTransportProblem):
    """
    A class that solves single-phase tracer transport problems by solving the
    exponents of the tracer concentration. This formulation guarantees the non-
    negativity of such concentration.
    """

    one = Constant(1.0)
    def initialize_form(self):
        """"""

        self.__u = TrialFunction(self.comp_func_spaces)
        self._TracerTransportProblem__w = TestFunction(self.comp_func_spaces)
        self.__w = self._TracerTransportProblem__w

        self._TracerTransportProblem__u = as_vector([exp(u) for u in self.__u])

        u = self._TracerTransportProblem__u

        self.tracer_forms = [Constant(0.0)*inner(self.__w, u)*self.dx]*super().num_forms
