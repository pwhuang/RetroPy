import sys
from problem_NaCl import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([1.33e-3, 2.03e-3, 9.31e-3, 5.28e-3,
                                        2.02e-3, 0.81e-3, 1.17e-3]) #mm^2/sec

problem = Problem(nx=200, ny=100, const_diff=False)
problem.generate_output_instance(sys.argv[1])

problem.define_problem()
problem.setup_flow_solver(r_val=5e6, omega_by_r=1.0)
problem.setup_reaction_solver()
problem.setup_auxiliary_reaction_solver()
problem.setup_transport_solver()

time_stamps = []
problem.solve(dt_val=1e0, endtime=900.0, time_stamps=time_stamps)
