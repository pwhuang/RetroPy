from dolfin import XDMFFile, MPI
import csv

class XDMFManager:
    def generate_output_instance(self, file_name: str):
        self.output_file_name = file_name

        self.outputter = XDMFFile(MPI.comm_world, file_name + '.xdmf')
        self.outputter.write(self.mesh)

        self.outputter.parameters['flush_output'] = True
        self.outputter.parameters['functions_share_mesh'] = True
        self.outputter.parameters['rewrite_function_mesh'] = False

        if MPI.rank(MPI.comm_world)==0:
            self.csv_file = open(self.output_file_name + '_time.csv', mode='w')
            self.csv_writer = csv.writer(self.csv_file, delimiter=',')

        return True

    def delete_output_instance(self):
        try:
            self.outputter
        except:
            return False

        self.outputter.close()
        return True

    def write_function(self, function, name, time_step):
        self.outputter.write_checkpoint(function, name,
                                        time_step=time_step,
                                        append=True)

    def flush_output(self):
        pass
