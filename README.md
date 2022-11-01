# RetroPy
## Environment
RetroPy uses FEniCS 2019.1.0 and Reaktoro v1. The environment can be installed using conda:
```
conda create -n fenics39 -c conda-forge fenics numpy scipy matplotlib jupyter reaktoro=1.2.3 python=3.9
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
