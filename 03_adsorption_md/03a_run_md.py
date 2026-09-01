import numpy as np
from ase.build import bulk
from ase.io import write, read
from ase.constraints import FixAtoms
from pathlib import Path
from mace_calculator import calc
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    thermalize_momenta,
    Stationary,
)
from ase.io.trajectory import Trajectory
from ase import units


# Slab parameters
a = 5.64       # NaCl lattice constant, Å
nx, ny = 3, 3  # lateral size
nz = 2         # conventional cells thick = 6 NaCl(001) atomic planes
vacuum = 8.0   # vacuum on each side, Å

# NVT simulation parameters
temperature_K = 400.0
timestep_fs = 0.5
friction_fs_inv = 0.002     # damping time = 500 fs
steps = 10_000_000          # 5 ns at 0.5 fs
write_interval = 100        # save every 50 fs

# Set the random number generator seed
rng = np.random.default_rng(123)

molecule_path = Path('../02_bulk_solution_md/HgCl2_minimization/HgCl2_minimized.xyz')

out_dir = Path('adsorption_md')
out_dir.mkdir(parents=True, exist_ok=True)

# Conventional cubic rocksalt cell: 4 NaCl formula units
unit = bulk(
    "NaCl",
    crystalstructure="rocksalt",
    a=a,
    cubic=True,
)

# (001) is normal to the conventional cell's z axis
slab = unit.repeat((nx, ny, nz))
slab.set_pbc((True, True, False))
slab.center(vacuum=vacuum, axis=2)
z_shift = -np.min(slab.positions[:, 2]) + 0.01
slab.translate((0, 0, z_shift))

# Fix the bottom two (001) planes
zmin = slab.positions[:, 2].min()
fixed = slab.positions[:, 2] < zmin + 0.51 * a
slab.set_constraint(FixAtoms(mask=fixed))

# Load and attach HgCl2 in the middle of the slab
molecule = read(molecule_path)
molecule.rotate(a=90, v=(1, 0, 0), center='COM')
molecule_offset = slab.get_center_of_mass() - molecule.get_center_of_mass()
molecule_offset_z = np.max(slab.positions[:, 2]) + 3.0 - np.min(molecule.positions[:, 2])
molecule.translate((molecule_offset[0], molecule_offset[1], molecule_offset_z))
slab.extend(molecule)

slab.info["charge"] = 0
slab.info["spin"] = 1
slab.info["external_field"] = [0.0, 0.0, 0.0]
slab.calc = calc


thermalize_momenta(
    slab,
    temperature_K=temperature_K,
    rng=rng,
    exact_temperature=False
)

# Remove initial net translational momentum
Stationary(slab, preserve_temperature=True)


dyn = Langevin(
    slab,
    timestep=timestep_fs * units.fs,
    temperature_K=temperature_K,
    friction=friction_fs_inv / units.fs,
    fixcm=False,
    rng=rng,
)

trajectory = Trajectory(
    out_dir / "nvt.traj",
    mode="w",
    atoms=slab,
)

logger = MDLogger(
    dyn,
    slab,
    out_dir / "nvt.log",
    header=True,
    stress=False,
    peratom=False,
    mode="w",
)

dyn.attach(trajectory.write, interval=write_interval)
dyn.attach(logger, interval=write_interval)


dyn.run(steps)

write(out_dir / "final.extxyz", slab)

print("Final temperature:", slab.get_temperature(), "K")
print("Final energy:", slab.get_potential_energy(), "eV")
