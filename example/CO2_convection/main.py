import sys
from problem import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([1.029e-3, 9.311e-3, 5.273e-3,
                                        2.045e-3, 0.804e-3, 1.101e-3]) #mm^2/sec

problem = Problem(nx=240, ny=96, const_diff=False)
problem.generate_output_instance(sys.argv[1])

problem.define_problem()
problem.setup_flow_solver(r_val=8e4, omega_by_r=1.0)
problem.setup_reaction_solver()
problem.setup_auxiliary_reaction_solver()
problem.setup_transport_solver()

time_stamps = [780.0]
problem.solve(dt_val=3e-2, endtime=1600.0, time_stamps=time_stamps)
