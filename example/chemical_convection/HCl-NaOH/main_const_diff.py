import sys
from problem import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([3.0e-3]*4) #mm^2/sec

problem = Problem(nx=31, ny=50)
problem.generate_output_instance(sys.argv[1])
problem.define_problem()

problem.setup_flow_solver()
problem.setup_reaction_solver()
#problem.setup_projection_solver()
problem.setup_transport_solver()

problem.setup_auxiliary_solver()

problem.solve(dt_val=1e-1, endtime=400.0)
