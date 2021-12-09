from . import *

class LoadOutputFile:
    def __init__(self, filename):
        self.filename = filename
        self.xdmf_obj = XDMFFile(filename + '.xdmf')

    def load_mesh(self):
        self.mesh = Mesh()
        self.xdmf_obj.read(self.mesh)

        return self.mesh

    def initialize_func_space(self, space, degree):
        self.func_space = FunctionSpace(self.mesh, space, degree)

    def load_function(self, func_name, timestep):
        func = Function(self.func_space)
        self.xdmf_obj.read_checkpoint(func, func_name, timestep)

        return func.copy()
