from ase.io import read, iread
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ase import units
from scipy.ndimage import uniform_filter1d
import pyblock
from scipy.stats import norm
from uncertainties import ufloat


trajectory_path = Path('bulk_solution_md/npt.traj')

AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660
TIME_STEP_FS  = 0.5
EQUIL_TIME_PS = 0.3

Hg_idx = 5
Cl_idxs = [4, 6]

# RDF settings
r_max = 8.0       # Å
n_bins = 200

bin_edges = np.linspace(0.0, r_max, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

shell_volumes = (
    4.0 / 3.0 * np.pi
    * (bin_edges[1:]**3 - bin_edges[:-1]**3)
)

rdf_sum = np.zeros(n_bins)
rdf_frames = 0

def get_hg_cl_coordination(atoms, r0=3.0, n=6):
    symbols = np.asarray(atoms.get_chemical_symbols())

    hg_indices = np.flatnonzero(symbols == "Hg")
    cl_indices = np.flatnonzero(symbols == "Cl")

    if len(hg_indices) == 0 or len(cl_indices) == 0:
        return np.zeros(len(hg_indices))

    # Create every Hg-Cl index pair
    hg_pairs = np.repeat(hg_indices, len(cl_indices))
    cl_pairs = np.tile(cl_indices, len(hg_indices))

    distances = atoms.get_distances(
        hg_pairs,
        cl_pairs,
        mic=True,
    ).reshape(len(hg_indices), len(cl_indices))

    switching_values = 1.0 / (1.0 + (distances / r0) ** n)

    # One coordination number per Hg atom
    return switching_values.sum(axis=1) * 2


def get_blocked_data(x, block_level):
    x = np.asarray(x)

    block_size = 2 ** block_level
    n_blocks = len(x) // block_size

    # Match pyblock: discard incomplete final block
    x = x[:n_blocks * block_size]

    return x.reshape(n_blocks, block_size).mean(axis=1)


def get_error_bars(x):
    # Documentation: https://pyblock.readthedocs.io/en/latest/tutorial.html

    reblock_data = pyblock.blocking.reblock(x)
    opt = pyblock.blocking.find_optimal_block(len(x), reblock_data)

    if np.isnan(opt[0]):
        print("Could not determine optimal block size; trajectory may be too short.")
        return ufloat(np.mean(x), 0.0)

    i = opt[0]
    result = reblock_data[i]

    block_size = 2 ** result.block
    mean = float(result.mean)
    stderr = float(result.std_err)

    blocked_x = get_blocked_data(x, result.block)

    return ufloat(mean, stderr), blocked_x, block_size

times = []
pressures = []
densities = []
coordination_nums = []
angles = []

for i, atoms in enumerate(iread(trajectory_path, index=":")):
    mass_amu = atoms.get_masses().sum()
    volume_A3 = atoms.get_volume()

    density = mass_amu / volume_A3 * AMU_PER_A3_TO_G_PER_CM3
    densities.append(density)

    times.append(i * TIME_STEP_FS)

    stress = atoms.get_stress(voigt=False, include_ideal_gas=True) / units.bar
    pressures.append(-np.mean(stress.diagonal()))

    coordination_nums.append(get_hg_cl_coordination(atoms, r0=2.5))

    # Calculate Cl-Hg-Cl angle
    theta = atoms.get_angle(Cl_idxs[0], Hg_idx, Cl_idxs[1], mic=True)
    angles.append(theta)

    # Compute RDF
    symbols = np.asarray(atoms.get_chemical_symbols())

    hg_indices = np.flatnonzero(symbols == "Hg")
    cl_indices = np.flatnonzero(symbols == "Cl")

    if len(hg_indices) == 0 or len(cl_indices) == 0:
        continue

        # Generate every Hg-Cl pair
    hg_pairs = np.repeat(hg_indices, len(cl_indices))
    cl_pairs = np.tile(cl_indices, len(hg_indices))

    distances = atoms.get_distances(
        hg_pairs,
        cl_pairs,
        mic=True,
    )

    histogram, _ = np.histogram(distances, bins=bin_edges)

    # Normalize using the Cl number density in this frame
    volume = atoms.get_volume()
    cl_number_density = len(cl_indices) / volume  # Cl/Å³

    ideal_counts = (
            len(hg_indices)
            * cl_number_density
            * shell_volumes
    )

    rdf_sum += histogram / ideal_counts
    rdf_frames += 1


# Time-averaged RDF
rdf = rdf_sum / rdf_frames

times = np.asarray(times) * 1e-3    # fs to ps
pressures = np.asarray(pressures)
densities = np.asarray(densities)
angles = np.asarray(angles)
coordination_nums = np.asarray(coordination_nums).squeeze()

start_idx = np.argmin(np.abs(times - EQUIL_TIME_PS))

ave_pressures, blocked_pressures, block_size_p = get_error_bars(pressures[start_idx:])
ave_densities, blocked_densities, block_size_rho = get_error_bars(densities[start_idx:])


fig, axs = plt.subplot_mosaic([
    ['A', 'B', 'C'],
    ['D', 'E', 'F'],
    ['.', '.', 'G']
], figsize=(14, 11), layout="constrained")

def plot_timeseries(ax, data, label, y_label, block_size=100, blocked_data=None, blocked_data_block_size=None):
    block_ave = uniform_filter1d(
        data,
        size=block_size,
        mode="reflect",
    )

    ax.plot(times, data, 'ob', markersize=0.5)
    ax.plot(times, block_ave, '-r')
    ax.axvline(times[start_idx], ls='--', color='black')
    ax.plot([times[start_idx]], [block_ave[start_idx]], 'or', markersize=8)
    # ax.axhline(ave_stress.n, ls='--', color='yellow')
    if blocked_data is not None:
        ax.plot(times[int(start_idx + blocked_data_block_size / 2):-int(blocked_data_block_size / 2):blocked_data_block_size], blocked_data, 'x', color='orange')
    ax.text(0.05, 0.95, label, transform=ax.transAxes,
            ha='left', va='top', weight='bold')

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.set_xlabel('Time, ps')
    ax.set_ylabel(y_label)

def plot_histogram(ax, blocked_data, ave_data, x_label):
    mu = np.mean(blocked_data)
    sigma = np.std(blocked_data, ddof=1)

    ax.hist(blocked_data, bins="auto", density=True,
            alpha=0.5, edgecolor="black", linewidth=1.0)

    xx = np.linspace(blocked_data.min(), blocked_data.max(), 200)
    ax.plot(xx, norm.pdf(xx, mu, sigma), '-k')
    ax.axvspan(ave_data.n-ave_data.s, ave_data.n+ave_data.s, alpha=0.2, color='blueviolet')
    ax.axvline(ave_data.n, ls='--', color='yellow', linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Normalized count')

plot_timeseries(axs['A'], pressures, 'Pressure', 'Pressure, bar', blocked_data=blocked_pressures, blocked_data_block_size=block_size_p)
plot_histogram(axs['B'], blocked_pressures, ave_pressures, x_label='Pressure, bar')
plot_timeseries(axs['D'], densities, 'Density', 'Density, g/cm^3', blocked_data=blocked_densities, blocked_data_block_size=block_size_rho)
plot_histogram(axs['E'], blocked_densities, ave_densities, x_label='Density, g/cm^3')

plot_timeseries(axs['C'], coordination_nums, 'Hg-Cl coordination number', 'Coordination number')
plot_timeseries(axs['G'], angles, 'Cl-Hg-Cl angle', 'Angle')
axs['F'].plot(bin_centers, rdf, color="tab:blue", linewidth=1.5)
axs['F'].set_xlabel('Hg-Cl distance, A')
axs['F'].set_ylabel('RDF')
axs['F'].text(0.05, 0.95, 'Hg-Cl RDF', transform=axs['F'].transAxes,
            ha='left', va='top', weight='bold')

fig.savefig('02_plot.png', dpi=300)

plt.show()
