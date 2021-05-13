from . import *

class TracerTransportProblem(TransportProblemBase):
    """A class that solves single-phase tracer transport problems."""

    def __init__(self):
        pass

    def set_num_component(self, num_component: int):
        self.num_component = num_component

    def set_component_fe_space(self, fe_space: str, fe_degree: int):
        self.FiniteElement = FiniteElement(fe_space,
                                           self.mesh.ufl_cell(), fe_degree)

        element_list = []
        for i in range(self.num_component):
            element_list.append(self.FiniteElement)

        self.comp_func_space = FunctionSpace(self.mesh, self.FiniteElement)
        self.comp_func_spaces = FunctionSpace(self.mesh,
                                              MixedElement(element_list))

        func_space_list = []
        for i in range(self.num_component):
            func_space_list.append(self.comp_func_space)

        self.assigner = FunctionAssigner(self.comp_func_spaces, func_space_list)

    def set_initial_condition(self, expression_list: list):
        if len(expression_list)!=self.num_component:
            raise Exception("length of expression_list != num_components")

        init_cond_list = []

        for expression in expression_list:
            init_cond_list.append(interpolate(expression,
                                              self.comp_func_space))

        self.u0 = Function(self.comp_func_spaces)
        self.assigner.assign(self.u0, init_cond_list)
