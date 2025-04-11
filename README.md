# RetroPy
[![DOI](https://zenodo.org/badge/196580566.svg)](https://zenodo.org/badge/latestdoi/196580566)
## Environment
RetroPy uses FEniCS-dolfinx 0.9.0 and Reaktoro v2. The environment can be installed using conda:
```
conda create -n fenics311 -c conda-forge fenics-dolfinx=0.9 numpy scipy h5py matplotlib jupyter reaktoro=2.12 python=3.11
conda activate fenics311
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
