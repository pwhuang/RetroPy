import sys
sys.path.insert(0, '../../')

from reaktoro_transport.problem import TracerTransportProblem, TracerTransportProblemExp
from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import TransientSolver, GradientSolver, TransientNLSolver

from numpy import zeros
from dolfin import Constant, Expression

def set_krylov_solver_params(prm):
    prm['absolute_tolerance'] = 1e-19
    prm['relative_tolerance'] = 1e-22
    prm['maximum_iterations'] = 500
    prm['error_on_nonconvergence'] = True
    prm['monitor_convergence'] = False
    prm['nonzero_initial_guess'] = True

class TransportManager(TracerTransportProblem, DG0Kernel, TransientSolver,
                       GradientSolver):

    def __init__(self, mesh, boundary_markers, domain_markers):
        TracerTransportProblem.__init__(self, mesh, boundary_markers, domain_markers)

    def set_component_properties(self):
        self.set_molecular_diffusivity([1.33e-3, 2.03e-3, 9.31e-3, 5.28e-3]) #mm^2/sec
        self.set_molar_mass([22.99, 35.453, 1.0, 17.0]) #g/mol
        self.set_solvent_molar_mass(18.0)
        self.set_charge([1.0, -1.0, 1.0, -1.0])

    def define_problem(self):
        self.set_components('Na+', 'Cl-', 'H+', 'OH-')
        self.set_solvent('H2O(l)')
        self.set_component_properties()

        self.set_component_fe_space()
        self.initialize_form()

        self.background_pressure = 101325 + 1e-3*9806.65*25 # Pa

        HCl_amounts = [1e-13, 1.0, 1.0, 1e-13, 54.17] # mmol/mm^3
        NaOH_amounts = [1.0, 1e-13, 1e-13, 1.0, 55.36]

        # HCl_amounts = [1e-10, 1e3, 1e3, 1e-10, 54.17e3] # mumol/mm^3
        # NaOH_amounts = [1e3, 1e-10, 1e-10, 1e3, 55.36e3]

        init_expr_list = []

        for i in range(self.num_component):
            init_expr_list.append('x[1]<=25.0 ?' + str(NaOH_amounts[i]) + ':' + str(HCl_amounts[i]))

        self.set_component_ics(Expression(init_expr_list, degree=1))
        self.set_solvent_ic(Expression('x[1]<=25.0 ?' + str(NaOH_amounts[-1]) + ':' + str(HCl_amounts[-1]) , degree=1))

        self.initialize_Reaktoro()
        self._set_temperature(298, 'K') # Isothermal problem

        self.num_dof = self.get_num_dof_per_component()
        self.rho_temp = zeros(self.num_dof)
        self.molar_density_temp = zeros([self.num_dof, self.num_component+1])

    def add_physics_to_form(self, u, kappa=Constant(1.0), f_id=0):
        self.set_advection_velocity()

        theta = Constant(1.0)
        one = Constant(1.0)

        self.add_explicit_advection(u, kappa=one, marker=0, f_id=f_id)
        #self.add_implicit_advection(kappa=theta, marker=0, f_id=f_id)

        #self.add_electric_field_advection(u, kappa=one, marker=0, f_id=f_id)

        for component in self.component_dict.keys():
            self.add_implicit_diffusion(component, kappa=theta, marker=0)
            self.add_explicit_diffusion(component, u, kappa=one-theta, marker=0)

        self.add_semi_implicit_charge_balanced_diffusion(u, kappa=theta, marker=0)
        self.add_explicit_charge_balanced_diffusion(u, kappa=one-theta, marker=0)
        #self.add_implicit_charge_balanced_diffusion(kappa=theta, marker=0)

    def setup_transport_solver(self):
        self.generate_solver()
        prm = self.set_solver_parameters('gmres', 'amg')
        #prm = self.get_solver_parameters()
        set_krylov_solver_params(prm['krylov_solver'])

    def get_components_min_value(self):
        return self.fluid_components.vector().min()
