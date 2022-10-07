from setuptools import setup

setup(
    name='RetroPy',
    version='0.0.1',
    description='An interface for reactive-transport simulations using FEniCS and Reaktoro',
    packages=['retropy', 'retropy.material',
              'retropy.mesh', 'retropy.physics',
              'retropy.problem', 'retropy.problem',
              'retropy.solver', 'retropy.tools',
              'retropy.manager'],
    package_dir={'': 'src'},
)
