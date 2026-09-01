from pathlib import Path

import numpy as np
from ase import units
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT
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
LX, LY, LZ = (20.0, 20.0, 12.0+5.0)

# Output directory
output_dir = Path("bulk_solution_md")
output_dir.mkdir(parents=True, exist_ok=True)

### Simulation parameters ###
temperature_K = 400.0           # Temperature
pressure_bar = 1.0              # Target hydrostatic pressure in bar
timestep_fs = 0.5               # Integration time step
thermostat_damping_fs = 50.0    # Thermostat damping time
friction_fs_inv = 1.0 / thermostat_damping_fs
barostat_damping_fs = 500.0     # Barostat damping time
nvt_steps = 10_000              # 5 ps at 0.5 fs
npt_steps = 1_000_000           # 500 ps at 0.5 fs
write_interval = 100            # Save dumps and statistics every 50 fs

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
dyn_nvt = Langevin(
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
    dyn_nvt,
    atoms,
    output_dir / "nvt.log",
    header=True,
    stress=False,
    peratom=False,
    mode="w",
)
dyn_nvt.attach(trajectory.write, interval=write_interval)
dyn_nvt.attach(logger, interval=write_interval)

### Run the NVT MD simulation ###
dyn_nvt.run(nvt_steps)

### Configure molecular dynamics in NPT ensemble ###
# IsotropicMTKNPT class performs integration, thermostating, and barostating
dyn_npt = IsotropicMTKNPT(
    atoms,
    timestep=timestep_fs * units.fs,
    temperature_K=temperature_K,
    pressure_au=pressure_bar * units.bar,
    tdamp=thermostat_damping_fs * units.fs,
    pdamp=barostat_damping_fs * units.fs,
)
# Trajectory class periodically writes atomic dumps
trajectory = Trajectory(
    output_dir / "npt.traj",
    mode="w",
    atoms=atoms,
)
# MDLogger class periodically writes statistics
logger = MDLogger(
    dyn_npt,
    atoms,
    output_dir / "npt.log",
    header=True,
    stress=True,
    peratom=False,
    mode="w",
)
dyn_npt.attach(trajectory.write, interval=write_interval)
dyn_npt.attach(logger, interval=write_interval)

### Run the NPT MD simulation ###
dyn_npt.run(npt_steps)

### Save the final configuration in EXTXYZ format (contains cell information and other stuff unlike XYZ) ###
write(output_dir / "final.extxyz", atoms)

### Output some final statistics ###
print("Final temperature:", atoms.get_temperature(), "K")
print("Final energy:", atoms.get_potential_energy(), "eV")
