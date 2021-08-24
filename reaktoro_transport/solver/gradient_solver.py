from . import *

class GradientSolver(ProjectionSolver):
    # def __init__(self, projection_space: FunctionSpace):
    #     super().__init__(projection_space)

    def set_projection_form(self, func_to_project: Function):
        V = self.projection_space

        w = TestFunction(V)
        u = TrialFunction(V)

        self.projection_form = inner(w, u)*self.dx
        self.projection_form += inner(div(w), func_to_project)*self.dx
        self.projection_form += -inner(self.n, w)*func_to_project*self.ds

    def generate_projection_solver(self, func_to_assign: Function, markers: list):
        a, L = lhs(self.projection_form), rhs(self.projection_form)
        bc_list = []
        for marker in markers:
            bc_list.append(DirichletBC(self.projection_space, Constant([0.0, 0.0]),
                                       self.boundary_markers, marker))

        problem = LinearVariationalProblem(a, L, func_to_assign, bcs=bc_list)
        self.projection_solver = LinearVariationalSolver(problem)
