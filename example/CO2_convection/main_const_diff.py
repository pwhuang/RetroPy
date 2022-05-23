import sys
from problem import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([3.0e-3]*self.num_component)

problem = Problem(nx=240, ny=96, const_diff=True)
problem.generate_output_instance(sys.argv[1])

problem.define_problem()
problem.setup_flow_solver(r_val=8e4, omega_by_r=1.0)
problem.setup_reaction_solver()
problem.setup_auxiliary_reaction_solver()
problem.setup_transport_solver()

time_stamps = [780.0]
problem.solve(dt_val=3e-2, endtime=1600.0, time_stamps=time_stamps)
