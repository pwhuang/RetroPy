from reaktoro_transport.problem import DarcyFlowUzawa
from dolfin import Constant, Function, info, PETScKrylovSolver, PETScLUSolver
from numpy import abs, max

def set_krylov_solver_params(prm):
    prm['absolute_tolerance'] = 1e-10
    prm['relative_tolerance'] = 1e-14
    prm['maximum_iterations'] = 8000
    prm['error_on_nonconvergence'] = True
    prm['monitor_convergence'] = False
    prm['nonzero_initial_guess'] = True

class FlowManager(DarcyFlowUzawa):
    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity(8.9e-4)  # Pa sec
        self.set_gravity([0.0, -9806.65]) # mm/sec

    def setup_flow_solver(self):
        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('RT', 1)

        self.set_pressure_ic(Constant(0.0))
        self._rho_old = Function(self.pressure_func_space)

        self.set_fluid_properties()

        self.mark_flow_boundary(pressure = [],
                                velocity = [self.marker_dict['top'], self.marker_dict['bottom'],
                                            self.marker_dict['left'], self.marker_dict['right']])

        self.set_pressure_bc([]) # Pa
        self.generate_form()
        #self.add_mass_source([-(self.fluid_density - self._rho_old)/self.dt])

        self.generate_residual_form()
        #self.add_mass_source_to_residual_form([-(self.fluid_density - self._rho_old)/self.dt])
        self.set_velocity_bc([Constant([0.0, 0.0])]*4)

        self.set_solver()
        self.set_additional_parameters(r_val=1e6, omega_by_r=1.0)
        self.assemble_matrix()

        # self.set_electric_field_form()
        # self.generate_electric_field_solver(markers=[1,2,3,4])
        # self.set_electric_field_solver_params()

    def solve_flow(self, target_residual: float, max_steps: int):
        # isSolved = False
        # r_val = 1e6
        # while isSolved==False:
        #     try:
        #         super().solve_flow(target_residual, max_steps)
        #         isSolved=True
        #     except:
        #         self.fluid_velocity.vector()[:] = 0.0
        #         r_val *= 1.1
        #         self.set_additional_parameters(r_val=r_val, omega_by_r=1.0)

        super().solve_flow(target_residual, max_steps)

        #info('Max velocity: ' + str( (self.fluid_velocity.vector().max() )))

    def set_solver(self, **kwargs):
        # Users can override this method.
        # Or, TODO: make this method more user friendly.

        #self.solver_v = PETScKrylovSolver('bicgstab', 'jacobi')
        self.solver_v = PETScLUSolver('mumps')
        self.solver_p = PETScKrylovSolver('gmres', 'jacobi')

        prm_v = self.solver_v.parameters
        prm_p = self.solver_p.parameters

        #set_krylov_solver_params(prm_v)
        set_krylov_solver_params(prm_p)
