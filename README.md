# rocoboom_out

**Ro**bust **co**ntrol using **boo**tstrapped **m**ultiplicative noise from **out**put measurements

## Package dependencies
- numpy
- scipy
- matplotlib
- control 
- cvxpy
- casadi
- [SIPPY](https://github.com/CPCLAB-UNIPI/SIPPY) (included as a submodule by this repo)

### Conda 

Run the package installation commands

```
conda install numpy scipy matplotlib
```

```
conda install -c conda-forge control
```

```
conda install -c conda-forge cvxpy
```

```
pip install casadi
```

## Installation

## Cloning the Package
The package contains submodules. Please clone with:
```
git clone [TODO URL].git --recurse
```
Note: When using GitHub desktop, the application initializes all submodules when cloning for the first time.
If a submodule is not initialized after pulling changes, please use the Git Bash tool or terminal and run the command

```
git submodule update --init
```

at the root of the repository.

## Usage

Run `monte_carlo.py` to see a lightweight test run using only 100 Monte Carlo trials. This should take a couple minutes at most to run and should produce several plots with various metrics (the same metrics depicted in the paper).
