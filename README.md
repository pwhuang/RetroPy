# RetroPy
[![DOI](https://zenodo.org/badge/196580566.svg)](https://zenodo.org/badge/latestdoi/196580566)
## Environment
RetroPy uses FEniCS-dolfinx 0.10.0 and Reaktoro v2. Navigate to the RetroPy folder. The environment can be installed using conda:
```
conda create -f environment.yml
conda activate fenicsx-env
```
We proceed with installing [Reaktoro](https://reaktoro.org/installation/installation-using-cmake.html)
```
git clone https://github.com/reaktoro/reaktoro.git
cd reaktoro
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
cmake --build build --parallel 4 --target install
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
cd tests
pytest
```
## Example Usage
To use RetroPy, we can execute one of the chemically-driven convection problem in the example folder:
```
cd example/chemical_convection/HCl-NaOH/
mpirun -n 4 python main.py output
```
where one can specify how many cpu cores to utilize after the -n option.
