from pathlib import Path
import numpy as np


from ase import Atoms
from ase.optimize import LBFGS, FIRE, BFGSLineSearch
from ase.io import write

from mace_calculator import calc

### Everything is the same here as in `01a_minimize_water_with_mace.py` except the molecule is initialized differently ###

### Atom positions are set explicitly because HgCl2 is an unknown molecule to ASE
r_hg_cl = 2.25  # initial Hg–Cl distance, Å
atoms = Atoms(
    symbols=["Cl", "Hg", "Cl"],
    positions=[
        [0.0, 0.0, -r_hg_cl],
        [0.0, 0.0,  0.0],
        [0.0, 0.0,  r_hg_cl],
    ],
)
atoms.center(vacuum=8.0)
atoms.set_pbc(False)

atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.info["external_field"] = [0.0, 0.0, 0.0]
atoms.calc = calc

out_dir = Path('HgCl2_minimization')
out_dir.mkdir(parents=True, exist_ok=True)

optimizer = LBFGS(
    atoms,
    trajectory=out_dir / "optimization.traj",
    logfile=out_dir / "optimization.log",
)
optimizer.run(fmax=0.01, steps=500)

write(out_dir / "HgCl2_minimized.xyz", atoms)

print("Energy:", atoms.get_potential_energy(), "eV")
print("Positions:\n", atoms.positions)
print("Maximum force:", abs(atoms.get_forces()).max(), "eV/Å")
