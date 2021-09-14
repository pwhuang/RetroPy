import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '../../')
from reaktoro_transport.mesh import MarkedRectangleMesh
from reaktoro_transport.problem import TracerTransportProblem, DarcyFlowMixedPoisson
from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import TransientSolver, GradientSolver

from dolfin import Constant, Expression, plot, Function, info
from dolfin.common.plotting import mplot_function
from numpy import zeros, shape, array
import matplotlib.pyplot as plt

def set_krylov_solver_params(prm):
    prm['absolute_tolerance'] = 1e-10
    prm['relative_tolerance'] = 5e-11
    prm['maximum_iterations'] = 8000
    prm['error_on_nonconvergence'] = False
    prm['monitor_convergence'] = True
    prm['nonzero_initial_guess'] = True

class ChemicallyDrivenConvection(TracerTransportProblem, DarcyFlowMixedPoisson,
                                 DG0Kernel, TransientSolver, GradientSolver):
    """
    """

    def __init__(self, nx, ny):
        TracerTransportProblem.__init__(self, *self.get_mesh_and_markers(nx, ny))

    def get_mesh_and_markers(self, nx, ny, mesh_type='triangle'):
        mesh_factory = MarkedRectangleMesh()
        mesh_factory.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        mesh_factory.set_top_right_coordinates(coord_x = 31.0, coord_y = 50.0)
        mesh_factory.set_number_of_elements(nx, ny)
        mesh_factory.set_mesh_type(mesh_type)

        mesh = mesh_factory.generate_mesh('crossed')
        boundary_markers, self.marker_dict = mesh_factory.generate_boundary_markers()
        domain_markers = mesh_factory.generate_domain_markers()

        return mesh, boundary_markers, domain_markers

    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_permeability(0.5**2/12.0) # mm^2
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity(8.9e-4)  # Pa sec
        self.set_gravity([0.0, -9806.65]) # mm/sec

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

        self.num_dof = len(self.fluid_components.vector()[:].reshape(-1, self.num_component))
        self.rho_temp = zeros(self.num_dof)
        self.molar_density_temp = zeros([self.num_dof, self.num_component+1])

        print(self.molar_density_temp.shape)
        self.background_pressure = 101325 + 1e-3*9806.65*25 # Pa

        HCl_amounts = [1e-16, 1e-3, 1e-3, 1e-16, 54.17e-3] # mol/mm^3
        NaOH_amounts = [1e-3, 1e-16, 1e-16, 1e-3, 55.36e-3]

        init_expr_list = []

        for i in range(self.num_component):
            init_expr_list.append('x[1]<=25.0 ?' + str(NaOH_amounts[i]) + ':' + str(HCl_amounts[i]))

        self.set_component_ics(Expression(init_expr_list, degree=1))
        self.set_solvent_ic(Expression('x[1]<=25.0 ?' + str(NaOH_amounts[-1]) + ':' + str(HCl_amounts[-1]) , degree=1))

        self.initialize_Reaktoro()
        self._set_temperature(298, 'K') # Isothermal problem

    def add_physics_to_form(self, u, kappa=Constant(1.0), f_id=0):
        theta = Constant(1.0)
        one = Constant(1.0)

        self.add_explicit_advection(u, kappa=theta, marker=0, f_id=f_id)

        for component in self.component_dict.keys():
            self.add_implicit_diffusion(component, kappa=theta, marker=0)
            #self.add_explicit_diffusion(component, u, kappa=one-theta, marker=0)

        self.add_semi_implicit_charge_balanced_diffusion(u, kappa=theta, marker=0)
        #self.add_explicit_charge_balanced_diffusion(u, kappa=one-theta, marker=0)

    def setup_flow_solver(self):
        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('RT', 1)
        self._rho_old = Function(self.pressure_func_space)

        self.set_fluid_properties()

        self.mark_flow_boundary(pressure = [],
                                velocity = [self.marker_dict['top'], self.marker_dict['bottom'],
                                            self.marker_dict['left'], self.marker_dict['right']])

        self.set_pressure_bc([]) # Pa
        DarcyFlowMixedPoisson.generate_form(self)
        #DarcyFlowMixedPoisson.add_mass_source(self, [-(self.fluid_density - self._rho_old)/self.dt])
        self.set_velocity_bc([Constant([0.0, 0.0])]*4)

        prm = self.set_solver('bicgstab', 'jacobi')
        set_krylov_solver_params(prm)
        self.set_additional_parameters(r_val=5e2)
        self.assemble_matrix()

    def setup_transport_solver(self):
        TransientSolver.generate_solver(self)
        TransientSolver.set_solver_parameters(self, 'gmres', 'amg')

    def _solve_chem_equi_over_dofs(self):
        pressure = self.fluid_pressure.vector()[:]
        component_molar_density = self.fluid_components.vector()[:].reshape(-1, self.num_component)
        solvent_molar_density = self.solvent.vector()[:]

        molar_density_list = []
        rho_list = []

        for i in range(self.num_dof):
            self._set_pressure(pressure[i] + self.background_pressure, 'Pa')
            self._set_species_amount(list(component_molar_density[i]) + [solvent_molar_density[i]])
            super().solve_chemical_equilibrium()
            self.molar_density_temp[i] = self._get_species_amounts()
            self.rho_temp[i] = self._get_fluid_density()*1e-6
            #molar_density_list.append(self._get_species_amounts())
            #rho_list.append(self._get_fluid_density()

        self.fluid_components.vector()[:] = self.molar_density_temp[:, :-1].flatten()
        self.fluid_density.vector()[:] = self.rho_temp

        #self.molar_density_temp = array(molar_density_list)
        #self.rho_temp = array(rho_list)*1e-6 # Convert to g/mm^3

    def solve_initial_condition(self):
        self._solve_chem_equi_over_dofs()
        self._rho_old.assign(self.fluid_density)

    def solve_chemical_equilibrium(self):
        self._solve_solvent_amount(self.get_solution())
        self._solve_chem_equi_over_dofs()

    def save_fluid_density(self, time):
        self.xdmf_obj.write_checkpoint(self.fluid_density,
                                       self.fluid_density.name(),
                                       time_step=time,
                                       append=True)

    def save_to_file(self, time):
        super().save_to_file(time, is_saving_pv=True)
        self.save_fluid_density(time)

        return True

    def solve(self, dt_val=1.0, timesteps=1):
        self.solve_initial_condition()
        self.solve_flow()

        self.set_dt(dt_val)

        current_time = 0.0
        self.save_to_file(time=current_time)

        for i in range(timesteps):
            current_time += dt_val
            info('timestep = ' + str(i+1) + ',  dt = ' + str(dt_val)\
                 + ', current_time = ' + str(current_time) )

            self.solve_one_step()
            self.assign_u1_to_u0()
            self.solve_chemical_equilibrium()
            self.solve_flow()

            self._rho_old.assign(self.fluid_density)
            self.save_to_file(time=current_time)

problem = ChemicallyDrivenConvection(nx=15, ny=30)
problem.generate_output_instance('chem_conv')
problem.define_problem()
problem.setup_flow_solver()
problem.setup_transport_solver()
problem.solve(dt_val=0.75, timesteps=350)

# vel, pres = problem.get_fluid_velocity(), problem.get_fluid_pressure()
# comp = problem.get_fluid_components()
# fig, ax = plt.subplots(2, 3, figsize=(12,8))
# mplot_function(ax[0,0], pres)
# mplot_function(ax[0,1], vel)
# mplot_function(ax[0,2], problem.fluid_density)
# mplot_function(ax[1,0], comp.sub(0))
# mplot_function(ax[1,1], comp.sub(1))
# cb = mplot_function(ax[1,2], comp.sub(2))
# fig.colorbar(cb)
# plt.show()
