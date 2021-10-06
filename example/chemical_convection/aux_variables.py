from dolfin import Function, TestFunction, assemble, Constant

class AuxVariables:
    def setup_auxiliary_solver(self):
        self.refractive_index = Function(self.DG0_space)
        self.refractive_index.rename('refractive_index', 'refractive_index')

        w = TestFunction(self.DG0_space)

        rho = self.fluid_density

        rho0 = 1e-3
        n0 = 1.333
        K = Constant((n0-1.0)/rho0) # Gladstone-Dale coefficient
        one = Constant(1.0)

        self.__form = w*(K*rho + one)/self.cell_volume*self.dx

    def solve_auxiliary_variables(self):
        self.refractive_index.vector()[:] = assemble(self.__form).get_local()

    def _save_auxiliary_variables(self, time):
        self.xdmf_obj.write_checkpoint(self.refractive_index,
                                       self.refractive_index.name(),
                                       time_step=time,
                                       append=True)
