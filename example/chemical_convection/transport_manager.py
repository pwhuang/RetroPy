import sys
sys.path.insert(0, '../../')

from reaktoro_transport.problem import TracerTransportProblemExp
from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import GradientSolver, TransientNLSolver

from dolfin import Constant, Expression, as_vector, info, derivative
from dolfin import MixedElement, FunctionSpace, Function

def set_krylov_solver_params(prm):
    prm['absolute_tolerance'] = 1e-13
    prm['relative_tolerance'] = 1e-12
    prm['maximum_iterations'] = 2000
    prm['error_on_nonconvergence'] = False
    prm['monitor_convergence'] = False
    prm['nonzero_initial_guess'] = True
    prm['divergence_limit'] = 1e6

class TransportManager(TracerTransportProblemExp, DG0Kernel, TransientNLSolver,
                       GradientSolver):

    def __init__(self, mesh, boundary_markers, domain_markers):
        TracerTransportProblemExp.__init__(self, mesh, boundary_markers, domain_markers)

    def add_physics_to_form(self, u, kappa=Constant(1.0), f_id=0):
        self.set_advection_velocity()

        theta = Constant(0.5)
        one = Constant(1.0)
        half = Constant(0.5)

        self.add_explicit_advection(u, kappa=one, marker=0, f_id=f_id)
        #self.add_implicit_advection(kappa=one, marker=0, f_id=f_id)
        #self.add_implicit_downwind_advection(kappa=half, marker=0, f_id=f_id)

        #self.add_electric_field_advection(u, kappa=one, marker=0, f_id=f_id)

        for component in self.component_dict.keys():
            self.add_implicit_diffusion(component, kappa=theta, marker=0)
            self.add_explicit_diffusion(component, u, kappa=one-theta, marker=0)

        self.add_implicit_charge_balanced_diffusion(kappa=one-theta, marker=0)
        #self.add_semi_implicit_charge_balanced_diffusion(u, kappa=one-theta, marker=0)
        self.add_explicit_charge_balanced_diffusion(u, kappa=theta, marker=0)

        self.evaluate_jacobian(self.get_forms()[0])

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        prm = self.get_solver_parameters()

        prm['nonlinear_solver'] = 'snes'

        nl_solver_type = 'snes_solver'

        prm[nl_solver_type]['absolute_tolerance'] = 1e-11
        prm[nl_solver_type]['relative_tolerance'] = 1e-13
        prm[nl_solver_type]['solution_tolerance'] = 1e-18
        prm[nl_solver_type]['maximum_iterations'] = 50
        prm['snes_solver']['method'] = 'newtonls'
        prm['snes_solver']['line_search'] = 'bt'
        prm['newton_solver']['relaxation_parameter'] = 0.1
        prm[nl_solver_type]['linear_solver'] = linear_solver
        prm[nl_solver_type]['preconditioner'] = preconditioner

        set_krylov_solver_params(prm[nl_solver_type]['krylov_solver'])
        #info(prm, True)

    def setup_transport_solver(self):
        TransientNLSolver.__init__(self)
        self.generate_solver(eval_jacobian=False)
        self.set_solver_parameters('gmres', 'amg')

    def setup_projection_solver(self):
        mixed_vel_func = []
        for i in range(self.num_component):
            mixed_vel_func.append(self.velocity_finite_element)

        projection_space = FunctionSpace(self.mesh, MixedElement(mixed_vel_func))

        self._grad_lna = Function(projection_space)
        GradientSolver.__init__(self, projection_space)

        self.set_projection_form(self.ln_activity)
        self.generate_projection_solver(self._grad_lna, [1,2,3,4])
        self.set_projection_solver_params()

    def get_components_min_value(self):
        return self.fluid_components.vector().min()
