from . import *

class TracerTransportProblem(TransportProblemBase):
    """A class that solves single-phase tracer transport problems."""

    def __init__(self):
        pass

    def set_components(self, *args):
        """Sets up the component dictionary.

        Input example: ['Na+', 'Cl-']
        """
        self.component_dict = {key: value for value, key in enumerate(args)}
        self.num_component = len(self.component_dict)

    def mark_component_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Keywords
        ---------
        {component_name : markers}
        Example: {'Na+': [1, 2, 3], 'outlet': [4]}
        """

        self.__boundary_dict = kwargs
        self.__dirichlet_bcs = []

    def set_component_fe_space(self):
        self.FiniteElement = FiniteElement(super().fe_space,
                                           self.mesh.ufl_cell(),
                                           super().fe_degree)

        element_list = []
        for i in range(self.num_component):
            element_list.append(self.FiniteElement)

        self.comp_func_spaces = FunctionSpace(self.mesh,
                                              MixedElement(element_list))

        self.func_space_list = []
        for i in range(self.num_component):
            self.func_space_list.append(self.comp_func_spaces.sub(i).collapse())

        self.assigner = FunctionAssigner(self.comp_func_spaces,
                                         self.func_space_list)

    def initialize_form(self):
        """"""

        self.__u = TrialFunction(self.comp_func_spaces)
        self.__w = TestFunction(self.comp_func_spaces)
        self.__u0 = Function(self.comp_func_spaces)
        self.__u1 = Function(self.comp_func_spaces)

        self.dt = Constant(1.0)

        self.__form = self.d_dt(self.__w, self.__u, self.__u0)*self.dx

    def set_component_ics(self, expressions: list):
        """"""

        if len(expressions)!=self.num_component:
            raise Exception("length of expressions != num_components")

        init_conds = []
        for i, expression in enumerate(expressions):
            init_conds.append(interpolate(expression,
                                              self.func_space_list[i]))

        self.assigner.assign(self.__u0, init_conds)

    def set_component_ic(self, component_name: str, expression):
        """"""
        #TODO: Make this function work.

        idx = self.component_dict[component_name]
        self.__u0[idx].assign(interpolate(expression, self.func_space_list[i]))

    def add_component_advection_bc(self, component_name: str, values):
        """"""

        if len(values)!=len(self.__boundary_dict[component_name]):
            raise Exception("length of values != number of markers")

        idx = self.component_dict[component_name]
        markers = self.__boundary_dict[component_name]

        for i, marker in enumerate(markers):
            self.__form += self.advection_flux_bc(self.__w[idx], values[i], marker)

    def add_outflow_bc(self):
        """"""

        for i, marker in enumerate(self.__boundary_dict['outlet']):
            self.__form += self.advection_outflow_bc(self.__w, self.__u, marker)

    def add_advection(self, marker: int):
        """Adds explicit advection physics to the variational form."""

        self.__form += self.advection(self.__w, self.__u0, marker)

    def add_diffusion(self, component_name: str, diffusivity, marker: int):
        """Adds implicit diffusion physics to the variational form."""

        self.__form += self.diffusion(self.__w, self.__u, diffusivity, marker)

    def add_dispersion(self):
        return #TODO: Setup this method.

    def generate_solver(self):
        """"""

        a, L = lhs(self.__form), rhs(self.__form)

        problem = LinearVariationalProblem(a, L, self.__u1, self.__dirichlet_bcs)
        self.__solver = LinearVariationalSolver(problem)

        prm = self.__solver.parameters
        prm['linear_solver'] = 'gmres'
        prm['preconditioner'] = 'ilu'

        TransportProblemBase.set_default_solver_parameters(prm['krylov_solver'])

    def solve_transport(self, dt_val: float, timesteps: int):
        """"""

        self.dt.assign(dt_val)
        self.save_to_file(time=0.0)

        for i in range(timesteps):
            self.__solver.solve()
            self.__u0.assign(self.__u1)
            self.save_to_file(time=(i+1)*dt_val)

        return self.__u0.copy()

    def save_to_file(self, time: float):
        """"""

        for key, i in self.component_dict.items():
            self.xdmf_obj.write_checkpoint(self.__u0.sub(i), key,
                                           time_step=time,
                                           append=True)
