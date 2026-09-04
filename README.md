# MACE for inorganic chemistry tutorial

### Software used and their respective documentation:
- ASE
- MACE-Polar (pytorch / ASE implementation)
- Packmol

### Useful commands:

`ase gui nvt.traj` - create an interactive visualization of an ASE trajectory

### Installing basic dependencies:

MACE and simulation environment:
```bash
pip install git+https://github.com/ACEsuit/mace.git@main
pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0
```

To accelerate on NVIDIA GPU:
```bash
pip install cuequivariance openequivariance
```

For data post-processing:
```bash
pip install pandas uncertainties pyblock
```

## Part 1: minimization of a molecule
