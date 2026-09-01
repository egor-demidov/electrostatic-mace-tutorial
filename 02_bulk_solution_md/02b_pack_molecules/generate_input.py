import numpy as np
from ase.io import read, write
import math

LX = 20.0
LY = 20.0
LZ = 12.0

# Thickness of the water slab
Z_PADDING = 5.0

LZ_T = LZ + Z_PADDING

RHO = 0.99  # g/cm^3
MW = 18.015 # g/mol
NA = 6.022e23   # 1/mol

vol = LX * LY * LZ * 1e-24  # cm^3
n_water = math.floor(RHO * vol / MW * NA)

print(f'Number of water molecules: {n_water}')
print(f'Effective density in cell with padding: {n_water / NA * MW / (LX * LY * LZ_T * 1e-24)}')


text = f"""
tolerance 2.0
filetype xyz
output bulk_solution_with_ions.xyz

seed -1
movebadrandom

# Simulation cell dimensions
pbc 0.0 0.0 0.0 {LX} {LY} {LZ_T}

structure Na.xyz
    number 1
    center
    fixed {LX/4} {LY/4} {LZ_T/2} 0.0 0.0 0.0
end structure

structure Cl.xyz
    number 1
    center
    fixed {LX*3/4} {LY/4} {LZ_T/2} 0.0 0.0 0.0
end structure

structure Na.xyz
    number 1
    center
    fixed {LX*3/4} {LY*3/4} {LZ_T/2} 0.0 0.0 0.0
end structure

structure Cl.xyz
    number 1
    center
    fixed {LX/4} {LY*3/4} {LZ_T/2} 0.0 0.0 0.0
end structure

structure ../HgCl2_minimization/HgCl2_minimized.xyz
    number 1
    center
    fixed {LX/2} {LY/2} {LZ_T/2} 0.0 0.0 0.0
end structure

# Read the minimized water and randomly insert N molecules
structure ../../01_molecule_minimization/water_minimization_mace/water_minimized.xyz
    number {n_water}
end structure
"""

with open('in.pack', 'w') as f:
    f.write(text)
