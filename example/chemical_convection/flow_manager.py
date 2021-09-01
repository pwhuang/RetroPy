import sys
sys.path.insert(0, '../../')

from reaktoro_transport.problem import DarcyFlowMixedPoisson
from dolfin import Constant, Function

def set_krylov_solver_params(prm):
    prm['absolute_tolerance'] = 1e-10
    prm['relative_tolerance'] = 1e-14
    prm['maximum_iterations'] = 8000
    prm['error_on_nonconvergence'] = True
    prm['monitor_convergence'] = False
    prm['nonzero_initial_guess'] = False

class FlowManager(DarcyFlowMixedPoisson):
    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_permeability(0.5**2/12.0) # mm^2
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity(8.9e-4)  # Pa sec
        self.set_gravity([0.0, -9806.65]) # mm/sec

    def setup_flow_solver(self):
        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('RT', 1)
        self._rho_old = Function(self.pressure_func_space)

        self.set_fluid_properties()

        self.mark_flow_boundary(pressure = [],
                                velocity = [self.marker_dict['top'], self.marker_dict['bottom'],
                                            self.marker_dict['left'], self.marker_dict['right']])

        self.set_pressure_bc([]) # Pa
        self.generate_form()
        #self.add_mass_source([-(self.fluid_density - self._rho_old)/self.dt])
        self.set_velocity_bc([Constant([0.0, 0.0])]*4)

        prm = self.set_solver('bicgstab', 'jacobi')
        set_krylov_solver_params(prm)
        self.set_additional_parameters(r_val=1e2)
        self.assemble_matrix()
