from pathlib import Path
import numpy as np
import argparse
from ase.io import read
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import pyblock
from uncertainties import ufloat
from scipy.stats import norm
import json


plt.rcParams.update({
    "font.size": 12,        # default text size
})

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


parser = argparse.ArgumentParser(
            prog='process_result')


parser.add_argument('work_dir', type=str, help='Directory containing outputs for the run')
parser.add_argument('--plot-block-size', type=int, default=100, help='Block size used for moving average in MC plots')
parser.add_argument('--equil-point', type=int, default=0, help='Index of point when the system is considered equilibrated')
parser.add_argument('--make-plot', action='store_true', help='Plot statistics')
parser.add_argument('--save-plot', action='store_true', help='Save plot in work_dir as stress_plot.png')

args = parser.parse_args()

work_dir = Path(args.work_dir)

data = np.loadtxt(work_dir / 'umbrella.log', delimiter=' ', skiprows=1)

pe = data[:, 1]
temp = data[:, 2]
force = data[:, 4]

time = np.arange(0, len(pe))
start_idx = args.equil_point

ave_pe, blocked_pe, block_size_pe = get_error_bars(pe[start_idx:])
ave_temp, blocked_temp, block_size_temp = get_error_bars(temp[start_idx:])
ave_force, blocked_force, block_size_force = get_error_bars(force[start_idx:])

print(f'Pxx: {ave_pe}')
print(f'Pyy: {ave_temp}')
print(f'Pzz: {ave_force}')

with open(work_dir / 'info.json', 'r') as file:
    json_data = json.load(file)
    restraint_r0 = json_data['r0_A']

# Write results to json

with open(work_dir / 'processed_stats.json', 'w') as file:
    json_data = {
        'r0_A': restraint_r0,
        'equil_point': args.equil_point,
        'pe_eV': {
            'mean': ave_pe.n,
            'error': ave_pe.s,
            'samples': len(blocked_pe),
            'block_size': block_size_pe
        },
        'temp_K': {
            'mean': ave_temp.n,
            'error': ave_temp.s,
            'samples': len(blocked_temp),
            'block_size': block_size_temp
        },
        'force_eV_A': {
            'mean': ave_force.n,
            'error': ave_force.s,
            'samples': len(blocked_force),
            'block_size': block_size_force
        }
    }

    json.dump(json_data, file, indent=4)

def plot_time_series(ax, stress, blocked_stress, block_size, ave_stress, label, ylabel):
    block_ave_stress = uniform_filter1d(
        stress,
        size=args.plot_block_size,
        mode="reflect",
    )

    ax.plot(time, stress, 'ob', markersize=0.5)
    ax.plot(time, block_ave_stress, '-r')
    ax.axvline(time[start_idx], ls='--', color='black')
    ax.plot([time[start_idx]], [block_ave_stress[start_idx]], 'or', markersize=8)
    try:
        ax.plot(time[int(start_idx+block_size/2):-int(block_size/2):block_size], blocked_stress, 'x', color='orange')
    except ValueError:
        pass
    ax.axhline(ave_stress.n, ls='--', color='yellow')
    ax.text(0.05, 0.95, label, transform=ax.transAxes,
            ha='left', va='top', weight='bold')

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.set_xlabel('Sample number')
    ax.set_ylabel(ylabel)

def plot_histogram(ax, blocked_stress, ave_stress, xlabel):
    mu = np.mean(blocked_stress)
    sigma = np.std(blocked_stress, ddof=1)

    ax.hist(blocked_stress, bins="auto", density=True,
            alpha=0.5, edgecolor="black", linewidth=1.0)

    xx = np.linspace(blocked_stress.min(), blocked_stress.max(), 200)
    ax.plot(xx, norm.pdf(xx, mu, sigma), '-k')
    ax.axvspan(ave_stress.n-ave_stress.s, ave_stress.n+ave_stress.s, alpha=0.2, color='blueviolet')
    ax.axvline(ave_stress.n, ls='--', color='yellow', linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Normalized count')

if args.make_plot:
    fig, axs = plt.subplot_mosaic([
        ['A', 'D'],
        ['B', 'E'],
        ['C', 'F']
    ], figsize=(9, 10), layout="constrained")

    plot_time_series(axs['A'], pe, blocked_pe, block_size_pe, ave_pe, 'PE', 'Potential energy, eV')
    plot_time_series(axs['B'], temp, blocked_temp, block_size_temp, ave_temp, 'T', 'Temperature, K')
    plot_time_series(axs['C'], force, blocked_force, block_size_force, ave_force, 'F', 'Force, eV/A')

    plot_histogram(axs['D'], blocked_pe, ave_pe, 'Potential energy, eV')
    plot_histogram(axs['E'], blocked_temp, ave_temp, 'Temperature, K')
    plot_histogram(axs['F'], blocked_force, ave_force, 'Force, eV/A')

    if args.save_plot:
        fig.savefig(work_dir / 'plot.png', dpi=300)

    plt.show()