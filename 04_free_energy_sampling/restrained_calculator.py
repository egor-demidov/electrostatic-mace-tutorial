import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.geometry import find_mic


class HarmonicDistanceBias(Calculator):
    """
    Harmonic umbrella potential:

        U(r) = 0.5 * k * (r - r0)**2

    Parameters
    ----------
    i, j : int
        Atom indices.
    r0 : float
        Umbrella center in Angstrom.
    k : float
        Spring constant in eV / Angstrom**2.
    mic : bool
        Use minimum-image convention. Usually False for a gas-phase cluster.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, i, j, r0, k, mic=False, **kwargs):
        super().__init__(**kwargs)
        self.i = int(i)
        self.j = int(j)
        self.r0 = float(r0)
        self.k = float(k)
        self.mic = bool(mic)

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        rij = atoms.positions[self.j] - atoms.positions[self.i]

        if self.mic:
            rij, r = find_mic(rij, atoms.cell, atoms.pbc)
            r = float(r)
        else:
            r = float(np.linalg.norm(rij))

        if r == 0.0:
            raise RuntimeError("Cannot evaluate distance restraint at r = 0.")

        dr = r - self.r0
        bias_energy = 0.5 * self.k * dr**2

        # rij points from atom i to atom j.
        force_on_i = self.k * dr * rij / r

        forces = np.zeros((len(atoms), 3))
        forces[self.i] += force_on_i
        forces[self.j] -= force_on_i

        self.results = {
            "energy": bias_energy,
            "forces": forces,
        }