# RetroPy
## Environment
RetroPy uses FEniCS 2019.1.0 and Reaktoro v1. The environment can be installed using conda:
```
conda create -n fenics39 -c conda-forge fenics numpy scipy h5py matplotlib jupyter reaktoro=1.2.3 python=3.9
conda activate fenics39
```
## Installation
For development purposes, please follow the procedure in the project directory:
```
python setup.py bdist_wheel
pip install -e .
```
## Testing
After installation, we can check whether it is correctly installed using pytest:
```
pip install pytest
cd $RetroPy/tests
pytest
```
## Example Usage
To use RetroPy, we can execute one of the chemically-driven convection problem in the example folder:
```
cd example/chemical_convection/HCl-NaOH/
mpirun -n 4 python main.py output
```
where one can specify how many cpu cores to utilize after the -n option.
