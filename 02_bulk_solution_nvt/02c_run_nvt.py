from pathlib import Path

import numpy as np
from ase import units
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    thermalize_momenta,
    Stationary,
)
from ase.optimize import FIRE
from ase.io.trajectory import Trajectory

from mace_calculator import calc

# Packed configuration
input_file = Path('02b_pack_molecules/bulk_solution_with_ions.xyz')

# Box dimensions, since XYZ files do not contain information about the box
LX, LY, LZ = (20.0, 20.0, 12.0)

# Output directory
output_dir = Path("bulk_solution_nvt")
output_dir.mkdir(parents=True, exist_ok=True)

### Simulation parameters ###
temperature_K = 400.0       # Temperature
timestep_fs = 0.5           # Integration time step
friction_fs_inv = 0.002     # Thermostat inverse damping time = 500 fs
steps = 40_000              # 20 ps at 0.5 fs
write_interval = 100        # Save dumps and statistics every 50 fs

### Seed the random number generator ###
rng = np.random.default_rng(1234)

### Read and configure the simulation box ###
atoms = read(input_file)
atoms.set_cell((LX, LY, LZ))        # Set bix dimensions
atoms.set_pbc((True, True, True))   # All boundaries are periodic

### Set the electronic parameters ###
atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.info["external_field"] = [0.0, 0.0, 0.0]
atoms.calc = calc

### Lightly minimize the system with FIRE to avoid huge overlaps ###
optimizer = FIRE(
    atoms,
    trajectory=output_dir / "minimization.traj",
    logfile=output_dir / "minimization.log",
    maxstep=0.1
)
optimizer.run(
    fmax=0.1,
    steps=500,
)

### Create temperature ###
thermalize_momenta(
    atoms,
    temperature_K=temperature_K,
    rng=rng,
    exact_temperature=False
)
# Remove initial net translational momentum
Stationary(atoms, preserve_temperature=True)

### Configure molecular dynamics in NVT ensemble ###
# Langevin class performs integration and thermostating
dyn = Langevin(
    atoms,
    timestep=timestep_fs * units.fs,
    temperature_K=temperature_K,
    friction=friction_fs_inv / units.fs,
    fixcm=False,
    rng=rng,
)
# Trajectory class periodically writes atomic dumps
trajectory = Trajectory(
    output_dir / "nvt.traj",
    mode="w",
    atoms=atoms,
)
# MDLogger class periodically writes statistics
logger = MDLogger(
    dyn,
    atoms,
    output_dir / "nvt.log",
    header=True,
    stress=False,
    peratom=False,
    mode="w",
)
dyn.attach(trajectory.write, interval=write_interval)
dyn.attach(logger, interval=write_interval)

### Run the MD simulation ###
dyn.run(steps)

### Save the final configuration in EXTXYZ format (contains cell information and other stuff unlike XYZ) ###
write(output_dir / "final.extxyz", atoms)

### Output some final statistics ###
print("Final temperature:", atoms.get_temperature(), "K")
print("Final energy:", atoms.get_potential_energy(), "eV")
