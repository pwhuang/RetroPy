from dolfin import HDF5File, MPI

class HDF5Manager:
    def generate_output_instance(self, file_name: str):
        self.output_file_name = file_name
        self.outputter = HDF5File(MPI.comm_world, file_name + '.h5', 'w')

        return True

    def delete_output_instance(self):
        try:
            self.outputter
        except:
            return False

        self.outputter.close()
        return True

    def write_function(self, function, name, time_step):
        self.outputter.write(function, name, t=time_step)

    def flush_output(self):
        self.outputter.flush()
