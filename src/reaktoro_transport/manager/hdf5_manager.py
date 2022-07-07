from dolfin import (HDF5File, MPI, vertex_to_dof_map, Function, MeshFunction,
                    XDMFFile)
import numpy as np
import csv

class HDF5Manager:
    def generate_output_instance(self, file_name: str):
        self.output_file_name = file_name
        self.outputter = HDF5File(MPI.comm_world, file_name + '.h5', 'w')
        self.outputter.write(self.mesh, 'mesh')
        xdmf_output = XDMFFile(MPI.comm_world, file_name + '_mesh.xdmf')
        xdmf_output.write(self.mesh)
        xdmf_output.close()

        if MPI.rank(MPI.comm_world)==0:
            self.csv_file = open(self.output_file_name + '.csv', mode='w')
            self.csv_writer = csv.writer(self.csv_file, delimiter=',')

        return True

    def delete_output_instance(self):
        try:
            self.outputter
            self.csv_file
        except:
            return False

        self.outputter.close()
        self.csv_file.close()
        return True

    def write_function(self, function, name, time_step):
        self.outputter.write(function, name, t=time_step)

    def flush_output(self):
        self.outputter.flush()

        if MPI.rank(MPI.comm_world)==0:
            self.csv_file.flush()

    def generate_input_instance(self, file_name: str):
        self.inputter = HDF5File(MPI.comm_world, file_name + '.h5', 'r')

    def read_function(self, *args):
        self.inputter.read(*args)
