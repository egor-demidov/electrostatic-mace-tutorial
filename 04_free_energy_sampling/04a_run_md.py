from mace_calculator import calc
from restrained_calculator import HarmonicDistanceBias
from ase.calculators.mixing import SumCalculator
from ase import Atoms
from ase.io import write, read
from pathlib import Path
import numpy as np
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    thermalize_momenta,
    Stationary,
)
from ase.io.trajectory import Trajectory
from ase import units
import argparse
from ase.constraints import FixCom
import json


parser = argparse.ArgumentParser()
parser.add_argument("--restraint-r0", type=float, required=True)
args = parser.parse_args()

vacuum = 15.0   # vacuum on each side, Å

# NVT simulation parameters
temperature_K = 300.0
timestep_fs = 0.5
friction_fs_inv = 0.002     # damping time = 500 fs
steps = 100_000             # 50 ps at 0.5 fs
write_interval = 100        # save every 50 fs
sample_interval = 50        # save every 25 fs

# Restraint parameters
restraint_r = args.restraint_r0
restraint_k = 1.0   # eV/A^2

# Set the random number generator seed
rng = np.random.default_rng(123)

hg_index = 1
cl_index = 3

molecule_path = Path('../02_bulk_solution_md/HgCl2_minimization/HgCl2_minimized.xyz')

out_dir = Path(f'free_energy_sampling_r0_{restraint_r:.02f}')
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / 'info.json', 'w') as file:
    info_obj = {
        'r0_A': restraint_r,
        'k_eV_A2': restraint_k
    }
    json.dump(info_obj, file, indent=4)

atoms = read(molecule_path)
atoms.center(vacuum=vacuum)
atoms.set_pbc((True, True, True))

chloride_pos = atoms.get_positions()[hg_index, :] + np.array((restraint_r, 0.0, 0.0))

chloride = Atoms(
    symbols=["Cl"],
    positions=[
        chloride_pos
    ]
)

atoms.extend(chloride)

umbrella = HarmonicDistanceBias(
    i=hg_index,
    j=cl_index,
    r0=restraint_r,       # Angstrom
    k=restraint_k,        # eV / Angstrom**2
    mic=True,
)

atoms.info["charge"] = -1
atoms.info["spin"] = 1
atoms.info["external_field"] = [0.0, 0.0, 0.0]
atoms.calc = SumCalculator([calc, umbrella])
atoms.set_constraint(FixCom())

thermalize_momenta(
    atoms,
    temperature_K=temperature_K,
    rng=rng,
    exact_temperature=False
)

# Remove initial net translational momentum
Stationary(atoms, preserve_temperature=True)

dyn = Langevin(
    atoms,
    timestep=timestep_fs * units.fs,
    temperature_K=temperature_K,
    friction=friction_fs_inv / units.fs,
    fixcm=False,
    rng=rng,
)

trajectory = Trajectory(
    out_dir / "nvt.traj",
    mode="w",
    atoms=atoms,
)

logger = MDLogger(
    dyn,
    atoms,
    out_dir / "nvt.log",
    header=True,
    stress=False,
    peratom=False,
    mode="w",
)

dyn.attach(trajectory.write, interval=write_interval)
dyn.attach(logger, interval=write_interval)

class UmbrellaLogger:
    def __init__(self, dyn, filename="umbrella.log"):
        self.dyn = dyn
        self.f = open(filename, "w")
        self.f.write("step pe temp delta_r force\n")

    def __call__(self):
        r = dyn.atoms.get_distance(hg_index, cl_index, vector=False, mic=True)
        delta_r = r - restraint_r
        force = restraint_k * delta_r

        self.f.write(
            f"{self.dyn.nsteps} "
            f"{self.dyn.atoms.get_potential_energy():.8f} "
            f"{self.dyn.atoms.get_temperature():.8f} "
            f"{delta_r:.8f} "
            f"{force:.8f}\n"
        )
        self.f.flush()

    def close(self):
        self.f.close()

umbrella_logger = UmbrellaLogger(dyn,  filename=out_dir / 'umbrella.log')
dyn.attach(umbrella_logger, interval=sample_interval)

dyn.run(steps)


