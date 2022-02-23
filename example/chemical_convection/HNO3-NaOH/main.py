import sys
from problem import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([1.334e-3, 1.902e-3, 9.311e-3, 5.273e-3]) #mm^2/sec

problem = Problem(nx=25, ny=90, const_diff=False)
problem.generate_output_instance(sys.argv[1])
problem.define_problem()

problem.setup_flow_solver()
problem.setup_reaction_solver()
problem.setup_transport_solver()

time_stamps = [3.0, 150.0, 700.0]
problem.solve(dt_val=1e-1, endtime=800.0, time_stamps=time_stamps)
