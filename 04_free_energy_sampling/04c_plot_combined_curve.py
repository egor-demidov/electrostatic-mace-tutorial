from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values, std_devs
from scipy.integrate import cumulative_trapezoid

plt.rcParams.update({
    "font.size": 12,        # default text size
})


dir_pattern = 'free_energy_sampling_r0_*.*/'

r0 = []
force = []

for subdir in Path('.').glob(dir_pattern):
    if (subdir / 'processed_stats.json').exists():
        with open(subdir / 'processed_stats.json', 'r') as f:
            data = json.load(f)

            r0.append(data['r0_A'])
            force.append(ufloat(data['force_eV_A']['mean'], data['force_eV_A']['error']))


r0 = np.asarray(r0)
force = np.asarray(force)

order = np.argsort(r0)

r0 = r0[order]
force = force[order]
delta_G = cumulative_trapezoid(-force, r0, initial=0)

# Shift the minimum of delta G to zero
delta_G -= min(delta_G)

fig, axs = plt.subplot_mosaic([
        ['A', 'B'],
], figsize=(9, 4), layout="constrained")

axs['A'].errorbar(r0, nominal_values(force), yerr=std_devs(force), ls='-', marker='o')
axs['B'].errorbar(r0, nominal_values(delta_G), yerr=std_devs(delta_G), ls='-', marker='o')
axs['A'].set_xlabel('r0, A')
axs['B'].set_xlabel('r0, A')
axs['A'].set_ylabel('f, eV/A')
axs['B'].set_ylabel('$\\Delta G$, eV')

plt.show()
