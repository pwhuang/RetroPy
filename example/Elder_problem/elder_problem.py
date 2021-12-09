import os
os.environ['OMP_NUM_THREADS'] = '1'

from reaktoro_transport.mesh import MarkedRectangleMesh
from reaktoro_transport.problem import TracerTransportProblem, DarcyFlowMixedPoisson
from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import TransientSolver

from dolfin import Constant, Function
from dolfin import SubDomain, near, DOLFIN_EPS
from ufl import as_vector

class MeshFactory(MarkedRectangleMesh):
    class BottomHalfBoundary(SubDomain):
        def __init__(self):
            super().__init__()

        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0, DOLFIN_EPS)\
                   and (x[0]<(3.0 + DOLFIN_EPS)) and (x[0]>(1.0 - DOLFIN_EPS))

    def __init__(self):
        super().__init__()

    def get_mesh_and_markers(self, nx, ny, mesh_type='triangle'):
        self.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        self.set_top_right_coordinates(coord_x = 4.0, coord_y = 1.0)
        self.set_number_of_elements(nx, ny)
        self.set_mesh_type(mesh_type)

        mesh = self.generate_mesh('crossed')

        boundary_markers, self.marker_dict = self.generate_boundary_markers()
        domain_markers = self.generate_domain_markers()

        return mesh, boundary_markers, domain_markers

    def generate_boundary_markers(self):
        super().generate_boundary_markers()

        bottom_half_marker = self.BottomHalfBoundary()
        bottom_half_marker.mark(self.boundary_markers, 5)

        marker_dict = {'right': 1, 'top': 2, 'left': 3,
                       'bottom_outer': 4, 'bottom_inner': 5}

        return self.boundary_markers, marker_dict

class FlowManager(DarcyFlowMixedPoisson):
    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_permeability(1.0)
        self.set_fluid_density(1.0)
        self.set_fluid_viscosity(1.0)
        self.set_gravity([0.0, 0.0])

    def setup_flow_solver(self):
        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('BDM', 1)

        self.set_fluid_properties()
        self.set_advection_velocity()

        self.mark_flow_boundary(pressure = [],
                                velocity = self.marker_dict.values())

        self.set_pressure_ic(Constant(0.0))
        self.set_pressure_bc([])
        self.generate_form()
        self.generate_residual_form()

        self.add_momentum_source([as_vector([Constant(0.0), self.fluid_components[0]])])
        self.add_momentum_source_to_residual_form([as_vector([Constant(0.0), self.fluid_components[0]])])
        self.set_velocity_bc([Constant([0.0, 0.0])]*5)

        prm = self.set_solver(solver_type='bicgstab', preconditioner='none')
        self.set_krylov_solver_params(prm)
        self.set_additional_parameters(r_val=5e-3, omega_by_r=1.0)
        self.assemble_matrix()

    def set_krylov_solver_params(self, prm):
        prm['absolute_tolerance'] = 1e-13
        prm['relative_tolerance'] = 1e-12
        prm['maximum_iterations'] = 8000
        prm['error_on_nonconvergence'] = True
        prm['monitor_convergence'] = True
        prm['nonzero_initial_guess'] = True

class TransportManager(TracerTransportProblem, DG0Kernel, TransientSolver):
    def __init__(self, nx, ny):
        TracerTransportProblem.__init__(self, *self.get_mesh_and_markers(nx, ny))

    def set_component_properties(self):
        self.set_molecular_diffusivity([1.0])

    def define_problem(self, Rayleigh_number=400.0):
        self.set_components('Temp')
        self.set_component_properties()
        self.mark_component_boundary(**{'Temp': [self.marker_dict['top'], self.marker_dict['bottom_inner']]})

        self.temp_bc = [Constant(0.0), Constant(1.0)]

        self.set_component_fe_space()
        self.initialize_form()

        self.set_component_ics(Constant([0.0, ]))
        self.Ra = Constant(Rayleigh_number)

    def add_physics_to_form(self, u, kappa=Constant(1.0), f_id=0):
        self.add_explicit_advection(u, kappa=self.Ra, marker=0, f_id=f_id)

        for component in self.component_dict.keys():
            self.add_implicit_diffusion(component, kappa=kappa, marker=0)
            self.add_component_diffusion_bc(component, Constant(5e2),\
                                            self.temp_bc, kappa, f_id)

    def setup_transport_solver(self):
        self.generate_solver()
        self.set_solver_parameters('gmres', 'amg')

    def solve_transport(self):
        self.solve_one_step()
        self.assign_u1_to_u0()

class ElderProblem(TransportManager, FlowManager, MeshFactory):
    """
    This is an example of solving the Elder problem using the Boussinesq approx-
    imation. Note that we set the fluid density to 1, and utilize the
    add_momentum_source method to apply the density term (a function of tempera-
    ture).
    """

    def __init__(self, nx, ny):
        TransportManager.__init__(self, nx, ny)
        self.define_problem()
        self.setup_flow_solver()
        self.setup_transport_solver()
        self.generate_output_instance('elder_problem')

    def solve(self, dt_val=1.0, timesteps=1):
        self.set_dt(dt_val)

        for i in range(timesteps):
            self.solve_transport()
            self.solve_flow(target_residual=5e-10, max_steps=5)

            self.save_to_file(time=(i+1)*dt_val)


problem = ElderProblem(nx=60, ny=15)
problem.solve(dt_val=2e-4, timesteps=50)
