# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx.io import XDMFFile
from mpi4py import MPI
import csv

class XDMFManager:
    def generate_output_instance(self, file_name: str):
        self.output_file_name = file_name

        self.outputter = XDMFFile(MPI.COMM_WORLD, file_name + '.xdmf', 'w')
        self.outputter.write_mesh(self.mesh)

        return True

    def delete_output_instance(self):
        try:
            self.outputter
        except:
            return False

        self.outputter.close()
        return True

    def write_function(self, function, time_step):
        self.outputter.write_function(function, t=time_step)

    def flush_output(self):
        pass
