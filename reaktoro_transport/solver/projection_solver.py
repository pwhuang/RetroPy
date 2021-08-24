from . import *

class ProjectionSolver:
    def __init__(self, projection_space: FunctionSpace):
        self.projection_space = projection_space

    def set_projection_form(self, func_to_project: Function):
        """L2 projection."""

        V = self.projection_space

        w = TestFunction(V)
        u = TrialFunction(V)

        self.projection_form = inner(w, u)*self.dx - inner(w, func_to_project)*self.dx

    def generate_projection_solver(self, func_to_assign: Function):
        a, L = lhs(self.projection_form), rhs(self.projection_form)
        problem = LinearVariationalProblem(a, L, func_to_assign, bcs=[])
        self.projection_solver = LinearVariationalSolver(problem)

    def set_projection_solver_params(self, linear_solver='gmres', preconditioner='jacobi'):
        prm = self.projection_solver.parameters
        prm['linear_solver'] = linear_solver
        prm['preconditioner'] = preconditioner

        prm_k = prm['krylov_solver']
        prm_k['absolute_tolerance'] = 1e-16
        prm_k['relative_tolerance'] = 1e-14
        prm_k['maximum_iterations'] = 5000
        prm_k['error_on_nonconvergence'] = True
        prm_k['monitor_convergence'] = False
        prm_k['nonzero_initial_guess'] = True

    def solve_projection(self):
        self.projection_solver.solve()
