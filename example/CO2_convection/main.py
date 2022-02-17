import sys
from problem import Problem

import matplotlib.pyplot as plt
from dolfin import plot

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([1.03e-3, 9.31e-3, 5.28e-3,
                                        2.02e-3, 0.81e-3, 1.17e-3]) #mm^2/sec

problem = Problem(filepath='mesh.xdmf', const_diff=False)
problem.generate_output_instance(sys.argv[1])

problem.define_problem()
problem.setup_flow_solver()
problem.setup_reaction_solver()
problem.setup_auxiliary_reaction_solver()
problem.setup_transport_solver()

time_stamps = [60.0, 360.0]
problem.solve(dt_val=1e-1, endtime=780.0, time_stamps=time_stamps)
