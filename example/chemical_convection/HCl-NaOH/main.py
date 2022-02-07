import sys
from problem import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([1.33e-3, 2.03e-3, 9.31e-3, 5.28e-3]) #mm^2/sec

problem = Problem(nx=62, ny=100, const_diff=False)
problem.generate_output_instance(sys.argv[1])
problem.define_problem()

problem.setup_flow_solver()
problem.setup_reaction_solver()
problem.setup_transport_solver()

time_stamps = [10.0, 40.0, 60.0, 115.0, 170.0, 200.0]
problem.solve(dt_val=1e-1, endtime=400.0, time_stamps=time_stamps)
