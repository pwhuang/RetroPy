import sys
from problem_NaCl import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([3.0e-3]*self.num_component)

problem = Problem(nx=200, ny=100, const_diff=False)
problem.generate_output_instance(sys.argv[1])

problem.define_problem()
problem.setup_flow_solver(r_val=5e6, omega_by_r=1.0)
problem.setup_reaction_solver()
problem.setup_auxiliary_reaction_solver()
problem.setup_transport_solver()

time_stamps = []
problem.solve(dt_val=1e0, endtime=900.0, time_stamps=time_stamps)
