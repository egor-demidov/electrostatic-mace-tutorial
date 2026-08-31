from pathlib import Path
import numpy as np

from ase.build import molecule                          # ASE function for building molecules
from ase.optimize import LBFGS, FIRE, BFGSLineSearch    # ASE minimization functions
from ase.io import write                                # ASE function for writing results

from mace_calculator import calc    # Our MACE calculator that evaluates forces and energies of a system of atoms

### Create a water molecule in vacuum ###
atoms = molecule("H2O")     # Create a molecule from a formula - works for common molecules
atoms.center(vacuum=8.0)    # Place the molecule in a box so that there are 8A of vacuum on each side
atoms.set_pbc(False)        # Turn off PBC in all directions

### Set electronic parameters - this part is similar to DFT ###
atoms.info["charge"] = 0    # Total charge of the system. Zero when molecules are neutral or each ion has a counterion
atoms.info["spin"] = 1      # Spin multiplicity - 2*S+1. S=0.5*N where N is the number of unpaired electrons
atoms.info["external_field"] = [0.0, 0.0, 0.0]  # External electric field applied to the box
atoms.calc = calc           # Attach the loaded calculator to our system of atoms

### Prepare a directory for output ###
out_dir = Path('water_minimization_mace')
out_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it does not already exist

### Perform optimization ###
# Different optimization algorithms are available in ASE.
#   - LBFGS: converges fast near the minimum, can be unstable if the initial guess is poor
#   - FIRE: handles poor initial configurations well, but can be slow to converge near the minimum
#   - BFGSLineSearch: guaranteed to decrease the target at each iteration. Can be slow, but will work where others fail
#   - More algorithms: https://docs.ase-lib.org/ase/optimize.html
# The algorithm can be switched by changing the line below:
optimizer = LBFGS(
    atoms,
    trajectory=out_dir / "optimization.traj",   # Atom dumps at each iteration
    logfile=out_dir / "optimization.log",       # Minimization statistics at each iteration
)
# Termination criteria:
#   - fmax: maximum force in the system
#   - steps: maximum number of optimizer iterations
optimizer.run(fmax=0.01, steps=500)

### Save the minimized water molecule in .xyz format (compatible with packmol) ###
write(out_dir / "water_minimized.xyz", atoms)

### Print some statistics ###
print("Energy:", atoms.get_potential_energy(), "eV")
print("Positions:\n", atoms.positions)
print("Maximum force:", abs(atoms.get_forces()).max(), "eV/Å")
