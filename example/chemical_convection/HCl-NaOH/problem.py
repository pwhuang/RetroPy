import os
os.environ['OMP_NUM_THREADS'] = '1'

from mesh_factory import MeshFactory
from reaktoro_transport.manager import DarcyFlowManagerUzawa as FlowManager
from reaktoro_transport.manager import ReactiveTransportManager

from dolfin import Expression, Constant

class Problem(ReactiveTransportManager, FlowManager, MeshFactory):
    """This class solves the chemically driven convection problem."""

    def __init__(self, nx, ny, const_diff):
        super().__init__(*self.get_mesh_and_markers(nx, ny))
        self.is_same_diffusivity = const_diff

    def set_component_properties(self):
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

        HCl_amounts = [1e-13, 1.0, 1.0, 1e-13, 54.17] # micro mol/mm^3 # mol/L
        NaOH_amounts = [1.0, 1e-13, 1e-13, 1.0, 55.36]

        init_expr_list = []

        for i in range(self.num_component):
            init_expr_list.append('x[1]<=25.0 ?' + str(NaOH_amounts[i]) + ':' + str(HCl_amounts[i]))

        self.set_component_ics(Expression(init_expr_list, degree=1))
        self.set_solvent_ic(Expression('x[1]<=25.0 ?' + str(NaOH_amounts[-1]) + ':' + str(HCl_amounts[-1]) , degree=1))

    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity(8.9e-4)  # Pa sec
        self.set_gravity([0.0, -9806.65]) # mm/sec
        self.set_permeability(0.5**2/12.0) # mm^2

    def set_flow_ibc(self):
        self.mark_flow_boundary(pressure = [],
                                velocity = [self.marker_dict['top'], self.marker_dict['bottom'],
                                            self.marker_dict['left'], self.marker_dict['right']])

        self.set_pressure_bc([]) # Pa
        self.set_pressure_ic(Constant(0.0))
        self.set_velocity_bc([Constant([0.0, 0.0])]*4)

    @staticmethod
    def timestepper(dt_val, current_time, time_stamp):
        min_dt, max_dt = 3e-2, 2.0

        if (dt_val := dt_val*1.1) > max_dt:
            dt_val = max_dt
        elif dt_val < min_dt:
            dt_val = min_dt
        if dt_val > time_stamp - current_time:
            dt_val = time_stamp - current_time

        return dt_val
