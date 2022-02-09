from setuptools import setup

setup(
    name='reaktoro_transport',
    version='0.0.1',
    description='An interface for reactive-transport simulations using FEniCS and Reaktoro',
    packages=['reaktoro_transport', 'reaktoro_transport.material',
              'reaktoro_transport.mesh', 'reaktoro_transport.physics',
              'reaktoro_transport.problem', 'reaktoro_transport.problem',
              'reaktoro_transport.solver', 'reaktoro_transport.tools',
              'reaktoro_transport.manager'],
    package_dir={'': 'src'},
)
