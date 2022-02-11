from . import *
import numpy as np

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

    def save_to_file(self, time: float, is_saving_pv=False):
        """"""

        try:
            self.xdmf_obj
        except:
            return False

        is_appending = True

        if self.num_component==1:
            self.output_assigner.assign(self.output_func_list[0], self.fluid_components)
        else:
            self.output_assigner.assign(self.output_func_list, self.fluid_components)

        for key, i in self.component_dict.items():
            self.output_func_list[i].vector()[:] = \
            np.exp(self.output_func_list[i].vector())

            self.xdmf_obj.write_checkpoint(self.output_func_list[i], key,
                                           time_step=time,
                                           append=is_appending)

        if is_saving_pv:
            self.save_fluid_pressure(time, is_appending)
            self.save_fluid_velocity(time, is_appending)

        return True

    def get_fluid_components(self):
        return as_vector([exp(self.fluid_components[i]) for i in range(self.num_component)])

    def set_component_ics(self, expressions: Expression):
        super().set_component_ics(expressions)

        if np.any(self.fluid_components.vector() < DOLFIN_EPS):
            raise ValueError('fluid_components contains negative or zero values!')

        self.fluid_components.vector()[:] = np.log(self.fluid_components.vector())
