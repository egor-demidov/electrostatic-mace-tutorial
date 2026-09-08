from ase.io import read, iread
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ase import units
from scipy.ndimage import uniform_filter1d, gaussian_filter
import pyblock
from scipy.stats import norm
from uncertainties import ufloat


trajectory_path = Path('adsorption_md/nvt.traj')

TIME_STEP_FS  = 0.5
EQUIL_TIME_PS = 0.5

Hg_idx = 145
Cl_idxs = [144, 146]

# Grid resolution
nx = 50
ny = 50

hg_histogram = np.zeros((nx, ny), dtype=float)
n_processed_frames = 0

# Use the first processed cell as the reference plotting cell
reference_lengths = None

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
pot_energies = []
bearings = []

for i, atoms in enumerate(iread(trajectory_path, index=":")):
    times.append(i * TIME_STEP_FS)
    pot_energies.append(atoms.get_potential_energy() / units.eV)

    # Mol vel
    mol_vec = atoms.get_distance(Cl_idxs[1], Cl_idxs[0], mic=True, vector=True)
    bearing = np.acos(mol_vec.dot((0.0, 1.0, 0.0)) / np.linalg.norm(mol_vec)) * 180.0 / np.pi
    bearings.append(bearing)

    symbols = np.asarray(atoms.get_chemical_symbols())
    hg_indices = np.flatnonzero(symbols == "Hg")

    # Initialize reference grid dimensions
    if reference_lengths is None:
        reference_lengths = atoms.cell.lengths()

    lx, ly = reference_lengths[:2]

    # Wrapped fractional coordinates in [0, 1)
    scaled_positions = atoms.get_scaled_positions(wrap=True)
    hg_scaled = scaled_positions[hg_indices]

    # Express positions in the reference XY cell
    hg_x = hg_scaled[:, 0] * lx
    hg_y = hg_scaled[:, 1] * ly

    frame_histogram, x_edges, y_edges = np.histogram2d(
        hg_x,
        hg_y,
        bins=(nx, ny),
        range=((0.0, lx), (0.0, ly)),
    )

    hg_histogram += frame_histogram
    n_processed_frames += 1


times = np.asarray(times) * 1e-3    # fs to ps
pot_energies = np.asarray(pot_energies)
bearings = np.asarray(bearings)

start_idx = np.argmin(np.abs(times - EQUIL_TIME_PS))

ave_pot_energies, blocked_pot_energies, block_size_pe = get_error_bars(pot_energies[start_idx:])

dx = x_edges[1] - x_edges[0]
dy = y_edges[1] - y_edges[0]
bin_area = dx * dy

# Units: Hg atoms / Å²
hg_density = hg_histogram / (n_processed_frames * bin_area)

print(
    "Integral of 2D density:",
    hg_density.sum() * bin_area,
)


fig, axs = plt.subplot_mosaic([
    ['A', 'B'],
    ['.', 'C'],
], figsize=(12, 9), layout="constrained")

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

# Plot lines at multiples of 45
axs['B'].axhline(45, ls='-', color='orange', lw=2)
axs['B'].axhline(90, ls='-', color='green', lw=2)
axs['B'].axhline(45*3, ls='-', color='orange', lw=2)
axs['B'].axhline(0, ls='-', color='green', lw=2)

plot_timeseries(axs['A'], pot_energies, 'Pot. energy', 'Pot. energy, eV', blocked_data=blocked_pot_energies, blocked_data_block_size=block_size_pe)
plot_timeseries(axs['B'], bearings, 'Bearing of HgCl2', 'Bearing of HgCl2, deg')

hg_density_smooth = gaussian_filter(
    hg_density,
    sigma=2.0,
    mode="wrap",  # preserves periodicity at the grid boundaries
)

mesh = axs['C'].pcolormesh(
    x_edges,
    y_edges,
    hg_density_smooth.T,
    shading="auto",
    cmap="magma",
)

axs['C'].set_xlabel(r"$x$ ($\mathrm{\AA}$)")
axs['C'].set_ylabel(r"$y$ ($\mathrm{\AA}$)")
axs['C'].set_aspect("equal")

colorbar = fig.colorbar(mesh, ax=axs['C'])
colorbar.set_label(r"Hg areal density ($\mathrm{\AA}^{-2}$)")

fig.savefig('03_plot.png', dpi=300)

plt.show()
