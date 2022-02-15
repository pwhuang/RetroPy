import os
os.environ['OMP_NUM_THREADS'] = '1'

from reaktoro_transport.problem import TracerTransportProblemExp
from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import TransientNLSolver

from benchmarks import ChargeBalancedDiffusion

from numpy import exp
from dolfin import Constant, as_vector, Function, norm
from dolfin.common.plotting import mplot_function
import matplotlib.pyplot as plt

def set_default_solver_parameters(prm):
    prm['absolute_tolerance'] = 1e-14
    prm['relative_tolerance'] = 1e-12
    prm['maximum_iterations'] = 5000
    prm['error_on_nonconvergence'] = True
    prm['monitor_convergence'] = False
    prm['nonzero_initial_guess'] = True

class DG0ExpChargeBalanceTest(TracerTransportProblemExp, DG0Kernel,
                              ChargeBalancedDiffusion, TransientNLSolver):
    def __init__(self, nx, t0, is_output=False):
        super().__init__(*self.get_mesh_and_markers(nx))

        self.set_flow_field()
        self.define_problem(t0=t0)

        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='jacobi')

        if is_output==True:
            self.generate_output_instance('charge_balance')

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        prm = self.get_solver_parameters()

        prm['nonlinear_solver'] = 'snes'

        nl_solver_type = 'snes_solver'

        prm[nl_solver_type]['absolute_tolerance'] = 1e-10
        prm[nl_solver_type]['relative_tolerance'] = 1e-14
        prm[nl_solver_type]['maximum_iterations'] = 50
        prm['snes_solver']['method'] = 'newtonls'
        prm['snes_solver']['line_search'] = 'bt'
        prm[nl_solver_type]['linear_solver'] = linear_solver
        prm[nl_solver_type]['preconditioner'] = preconditioner

        set_default_solver_parameters(prm[nl_solver_type]['krylov_solver'])

    def set_advection_velocity(self):
        E = self.electric_field
        D = self.molecular_diffusivity
        z = self.charge

        self.advection_velocity = \
        as_vector([self.fluid_velocity + z[i]*D[i]*E for i in range(self.num_component)])

    def add_physics_to_form(self, u):
        super().add_physics_to_form(u)

        theta = Constant(0.5)
        one = Constant(1.0)

        self.add_explicit_charge_balanced_diffusion(u, kappa=one-theta, marker=0)
        #self.add_semi_implicit_charge_balanced_diffusion(u, kappa=theta, marker=0)
        self.add_implicit_charge_balanced_diffusion(kappa=theta, marker=0)

    def solve_transport(self, dt_val=1.0, timesteps=1):
        self.set_dt(dt_val)
        self.save_to_file(time=0.0)

        for i in range(timesteps):
            self.solve_one_step()
            self.assign_u1_to_u0()
            self.save_to_file(time=(i+1)*dt_val)

    def get_error_norm(self):
        self.output_func = Function(self.comp_func_spaces)
        self.output_func.vector()[:] = exp(self.fluid_components.vector())

        mass_error = Function(self.comp_func_spaces)
        mass_error.assign(self.output_func - self.solution)

        mass_error_norm = norm(mass_error, 'l2')

        return mass_error_norm

    def mpl_output(self):
        fig, ax = plt.subplots(1,1)
        mplot_function(ax, self.solution.sub(0), lw=3, c='C0')
        mplot_function(ax, self.output_func.sub(0), ls=(0,(5,5)), lw=3, c='C1')
        mplot_function(ax, self.output_func.sub(1), ls=(0,(5,5)), lw=2, c='C3')
        plt.show()

nx, t0 = 51, 1.0
list_of_dt = [3e-1]
timesteps = [10]
err_norms = []

for i, dt in enumerate(list_of_dt):
    problem = DG0ExpChargeBalanceTest(nx, t0, is_output=False)
    problem.solve_transport(dt_val=dt, timesteps=timesteps[i])

    t_end = timesteps[i]*dt + t0
    problem.get_solution(t_end)
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)

print(err_norms)

def test_function():
    assert err_norms[-1] < 1e-2
