import sys
from problem import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([1.33e-3, 1.90e-3, 9.31e-3, 5.28e-3]) #mm^2/sec

problem = Problem()
problem.generate_output_instance(sys.argv[1])
problem.define_problem()

problem.setup_flow_solver()
problem.setup_reaction_solver()
problem.setup_projection_solver()
problem.setup_transport_solver()

problem.setup_auxiliary_solver()

problem.solve(dt_val=7e-1, endtime=1000.0, time_stamps=[1.0, 2.0, 2.005])
