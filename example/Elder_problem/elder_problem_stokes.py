import os
os.environ['OMP_NUM_THREADS'] = '1'

from reaktoro_transport.mesh import MarkedRectangleMesh, XDMFMesh
from reaktoro_transport.problem import TracerTransportProblem, StokesFlowUzawa
from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import TransientSolver
from reaktoro_transport.manager import HDF5Manager as OutputManager

from dolfin import (Constant, Function, MPI, SubDomain, near, DOLFIN_EPS,
                    MeshFunction, PETScLUSolver, PETScKrylovSolver,
                    VectorElement, FunctionSpace)
from ufl import as_vector
import numpy as np

class MeshFactory(MarkedRectangleMesh, XDMFMesh):
    class BottomHalfBoundary(SubDomain):
        def __init__(self, boundary_eps=1e-8):
            super().__init__()
            self.boundary_eps = boundary_eps

        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0, self.boundary_eps)\
                   and (x[0]<(3.0 + DOLFIN_EPS)) and (x[0]>(1.0 - DOLFIN_EPS))

    def __init__(self):
        super().__init__()

    def get_mesh_and_markers(self, filepath):
        self.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        self.set_top_right_coordinates(coord_x = 4.0, coord_y = 1.0)

        mesh = self.read_mesh(filepath)

        boundary_markers, self.marker_dict = self.generate_boundary_markers()
        domain_markers = self.generate_domain_markers()

        return mesh, boundary_markers, domain_markers

    def generate_boundary_markers(self, boundary_eps=1e-8):
        self.boundary_markers = MeshFunction('size_t', self.mesh,
                                             dim=self.mesh.geometric_dimension()-1)

        self.boundary_markers.set_all(0)

        self.AllBoundary().mark(self.boundary_markers, 555)
        self.RightBoundary(self.xmax, boundary_eps).mark(self.boundary_markers, 1)
        self.TopBoundary(self.ymax, boundary_eps).mark(self.boundary_markers, 2)
        self.LeftBoundary(self.xmin, boundary_eps).mark(self.boundary_markers, 3)
        self.BottomBoundary(self.ymin, boundary_eps).mark(self.boundary_markers, 4)

        bottom_half_marker = self.BottomHalfBoundary()
        bottom_half_marker.mark(self.boundary_markers, 5)

        marker_dict = {'right': 1, 'top': 2, 'left': 3,
                       'bottom_outer': 4, 'bottom_inner': 5, 'pores': 555}

        return self.boundary_markers, marker_dict

class FlowManager(StokesFlowUzawa):
    def set_fluid_properties(self):
        self.set_fluid_density(1.0)
        self.set_fluid_viscosity(1.0)
        self.set_gravity([0.0, 0.0])

    def setup_flow_solver(self):
        self.set_pressure_fe_space('DG', 1)
        self.set_velocity_vector_fe_space('CG', 2)

        self.set_fluid_properties()
        self.set_advection_velocity()

        self.generate_form()

        self.mark_flow_boundary(inlet = [], noslip = self.marker_dict.values(),
                                velocity_bc = [])

        self.set_pressure_ic(Constant(0.0))
        self.set_pressure_bc([])
        self.set_velocity_bc([])

        self.add_momentum_source([as_vector([Constant(0.0), self.fluid_components[0]])])
        self.add_momentum_source_to_residual_form([as_vector([Constant(0.0), self.fluid_components[0]])])

        self.set_flow_solver_params()
        self.set_additional_parameters(r_val=5e3, omega_by_r=1.0)
        self.assemble_matrix()

    def save_fluid_velocity(self, time_step):
        self.write_function(self.fluid_velocity, self.fluid_velocity.name(),
                            time_step)

    def set_flow_solver_params(self):
        self.solver_v = PETScLUSolver('mumps')
        self.solver_p = PETScKrylovSolver('gmres', 'jacobi')

        prm_v = self.solver_v.parameters
        prm_p = self.solver_p.parameters

        self.set_krylov_solver_params(prm_p)

    def set_krylov_solver_params(self, prm):
        prm['absolute_tolerance'] = 1e-13
        prm['relative_tolerance'] = 1e-12
        prm['maximum_iterations'] = 8000
        prm['error_on_nonconvergence'] = True
        prm['monitor_convergence'] = False
        prm['nonzero_initial_guess'] = True

class TransportManager(TracerTransportProblem, DG0Kernel, TransientSolver):
    def __init__(self, filepath):
        TracerTransportProblem.__init__(self, *self.get_mesh_and_markers(filepath))

    def set_component_properties(self):
        self.set_molecular_diffusivity([1.0])

    def define_problem(self, Rayleigh_number=5e5):
        self.set_components('Temp')
        self.set_component_properties()
        self.mark_component_boundary(**{'Temp': [self.marker_dict['top'], self.marker_dict['bottom_inner']]})

        self.temp_bc = [Constant(0.0), Constant(1.0)]

        self.set_component_fe_space()
        self.initialize_form()

        self.set_component_ics(Constant([0.0, ]))
        self.Ra = Constant(Rayleigh_number)

    def add_physics_to_form(self, u, kappa=Constant(0.5), f_id=0):
        self.add_explicit_advection(u, kappa=kappa*self.Ra, marker=0, f_id=f_id)
        self.add_implicit_advection(kappa=kappa*self.Ra, marker=0, f_id=f_id)

        for component in self.component_dict.keys():
            self.add_explicit_diffusion(component, u, kappa=kappa, marker=0)
            self.add_implicit_diffusion(component, kappa=kappa, marker=0)
            self.add_component_diffusion_bc(component, Constant(5e4),\
                                            self.temp_bc, kappa, f_id)

    def setup_transport_solver(self):
        self.generate_solver()
        self.set_solver_parameters('gmres', 'amg')

    def solve_transport(self):
        self.solve_one_step()
        self.assign_u1_to_u0()

class ElderProblem(TransportManager, FlowManager, MeshFactory, OutputManager):
    """
    This is an example of solving the Elder problem using the Boussinesq approx-
    imation. Note that we set the fluid density to 1, and utilize the
    add_momentum_source method to apply the density term (a function of tempera-
    ture).
    """

    def __init__(self, filepath):
        TransportManager.__init__(self, filepath)
        self.define_problem()
        self.setup_flow_solver()
        self.setup_transport_solver()
        self.generate_output_instance('elder_problem_stokes')

    def solve(self, dt_val=1.0, timesteps=1):
        self.set_dt(dt_val)
        saved_times = []

        for i in range(timesteps):
            self.solve_transport()
            self.solve_flow(target_residual=5e-10, max_steps=20)

            self.save_to_file(time=(i+1)*dt_val, is_saving_pv=True)
            saved_times.append((i+1)*dt_val)

        if MPI.rank(MPI.comm_world)==0:
            np.save(self.output_file_name + '_time', np.array(saved_times), allow_pickle=False)


problem = ElderProblem(filepath='pore_mesh.xdmf')
problem.solve(dt_val=3e-3, timesteps=50)
