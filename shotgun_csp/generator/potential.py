# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from ase.stress import full_3x3_to_voigt_6_stress

from shotgun_csp.utils import preset


class HookesLawCalculator(Calculator):
    """
    A calculator for computing energy, forces, and stress based on Hooke's Law.

    Attributes:
        implemented_properties (list): List of properties that the calculator can compute.
        k (float): Spring constant for Hooke's Law. Default is 1.0.
        s (float): Scaling factor for covalent radius. Default is 2.
        covalent_radius (dict): Dictionary of covalent radii for different elements. If not provided, a preset value is used.

    Methods:
        __init__(k=1.0, s=2, covalent_radius=None, **kwargs):
            Initializes the HookesLawCalculator with the given parameters.

        calculate(atoms=None, properties=["energy"], system_changes=all_changes):
            Computes the energy, forces, and stress for the given atomic configuration.
            Raises a ValueError if the atoms object is None or if positions, symbols, or cell are not properly initialized.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, k=1.0, s=2, covalent_radius=None, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.s = s
        self.covalent_radius = covalent_radius or preset.covalent_radius

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("The atoms object is None.")

        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        cell = atoms.get_cell()

        if positions is None or symbols is None or cell is None:
            raise ValueError("The positions, symbols, or cell are not properly initialized.")

        natoms = len(positions)
        energy = 0.0
        forces = np.zeros((natoms, 3))
        stress = np.zeros((3, 3))

        for i in range(natoms):
            for j in range(i + 1, natoms):
                r_ij = positions[i] - positions[j]
                r_ij = r_ij - np.dot(
                    np.round(np.dot(r_ij, np.linalg.inv(cell))), cell
                )  # Apply periodic boundary conditions
                r = np.linalg.norm(r_ij)
                d_ij = self.s * (self.covalent_radius.get(symbols[i], 0.0) + self.covalent_radius.get(symbols[j], 0.0))
                r_adjusted = r - d_ij
                energy += 0.5 * self.k * r_adjusted**2
                force_magnitude = -self.k * r_adjusted
                force = force_magnitude * (r_ij / r)  # normalize and scale the force
                forces[i] += force
                forces[j] -= force

                # Calculate stress contribution
                stress_contribution = np.outer(force, r_ij) / cell.volume
                stress += stress_contribution

        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = stress


class LennardJones(Calculator):
    """
    Lennard-Jones potential calculator.

    This class implements the Lennard-Jones potential for calculating
    energies, forces, and stresses in a system of atoms.

    Attributes:
        implemented_properties (list): List of properties that can be calculated.
        default_parameters (dict): Default parameters for the Lennard-Jones potential.
        nolabel (bool): Indicates whether the calculator has a label.

    Methods:
        __init__(**kwargs):
            Initializes the Lennard-Jones calculator with given parameters.

        calculate(atoms=None, properties=None, system_changes=all_changes):
            Calculates the specified properties for the given atoms.
            Parameters:
                atoms (Atoms): The atomic configuration.
                properties (list): List of properties to calculate.
                system_changes (list): List of changes in the system.
    """

    implemented_properties = ["energy", "energies", "forces", "free_energy"]
    implemented_properties += ["stress", "stresses"]  # bulk properties
    default_parameters = {
        "epsilon": 1.0,
        "sigma": 1.0,
        "dist_scale": 1.5,
        "rc": None,
        "ro": None,
        "smooth": False,
    }
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

        if self.parameters.rc is None:
            self.parameters.rc = 3 * self.parameters.sigma

        if self.parameters.ro is None:
            self.parameters.ro = 0.66 * self.parameters.rc

        self.nl = None

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)

        # 找到非零的最小距离
        distances = atoms.get_all_distances()
        min_distance = np.min(distances[distances > 0])
        target_distance = (
            np.max([preset.covalent_radius[e] for e in atoms.get_chemical_symbols()]) * self.parameters.dist_scale
        )

        force_scale = 1 if min_distance < target_distance else 0.001

        #         print(f'min_distance_current: {min_distance}')
        #         print(f'target_distance: {target_distance}')
        #         print(f'force_scale: {force_scale}')

        sigma = min_distance * (1 + min((target_distance / min_distance), 0.05))
        epsilon = self.parameters.epsilon
        rc = self.parameters.rc
        ro = self.parameters.ro
        smooth = self.parameters.smooth

        if self.nl is None or "numbers" in system_changes:
            self.nl = NeighborList([rc / 2] * natoms, self_interaction=False, bothways=True)

        self.nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell

        # potential value at rc
        e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        for ii in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)

            # pointing *towards* neighbours
            distance_vectors = positions[neighbors] + cells - positions[ii]

            r2 = (distance_vectors**2).sum(1)
            c6 = (sigma**2 / r2) ** 3
            c6[r2 > rc**2] = 0.0
            c12 = c6**2

            if smooth:
                cutoff_fn = cutoff_function(r2, rc**2, ro**2)
                d_cutoff_fn = d_cutoff_function(r2, rc**2, ro**2)

            pairwise_energies = 4 * epsilon * (c12 - c6)
            pairwise_forces = -24 * epsilon * (2 * c12 - c6) / r2  # du_ij

            if smooth:
                # order matters, otherwise the pairwise energy is already
                # modified
                pairwise_forces = cutoff_fn * pairwise_forces + 2 * d_cutoff_fn * pairwise_energies
                pairwise_energies *= cutoff_fn
            else:
                pairwise_energies -= e0 * (c6 != 0.0)

            pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors

            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies
            forces[ii] += pairwise_forces.sum(axis=0) * force_scale

            stresses[ii] += 0.5 * np.dot(pairwise_forces.T, distance_vectors)  # equivalent to outer product

        # no lattice, no stress
        if self.atoms.cell.rank == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results["stress"] = stresses.sum(axis=0) / self.atoms.get_volume()
            self.results["stresses"] = stresses / self.atoms.get_volume()

        energy = energies.sum()
        self.results["energy"] = energy
        self.results["energies"] = energies

        self.results["free_energy"] = energy

        self.results["forces"] = forces


def cutoff_function(r, rc, ro):
    """
    Computes the cutoff function value based on the given distances.

    The cutoff function is defined piecewise:
    - If r < ro, the function returns 1.0.
    - If ro <= r < rc, the function returns a value based on a polynomial expression.
    - If r >= rc, the function returns 0.0.

    Parameters:
    r (float or np.ndarray): The distance(s) at which to evaluate the cutoff function.
    rc (float): The cutoff distance beyond which the function value is 0.
    ro (float): The distance below which the function value is 1.

    Returns:
    float or np.ndarray: The value of the cutoff function at the given distance(s).
    """
    return np.where(
        r < ro,
        1.0,
        np.where(r < rc, (rc - r) ** 2 * (rc + 2 * r - 3 * ro) / (rc - ro) ** 3, 0.0),
    )


def d_cutoff_function(r, rc, ro):
    """
    Computes the cutoff function value based on the given distances.

    The function returns 0.0 if the distance `r` is less than `ro`. If `r` is between `ro` and `rc`,
    it computes the value using the formula: 6 * (rc - r) * (ro - r) / (rc - ro) ** 3. For `r` greater
    than or equal to `rc`, the function returns 0.0.

    Parameters:
    r (float or np.ndarray): The distance(s) at which to evaluate the cutoff function.
    rc (float): The cutoff distance beyond which the function value is 0.0.
    ro (float): The reference distance below which the function value is 0.0.

    Returns:
    float or np.ndarray: The computed cutoff function value(s).
    """
    return np.where(
        r < ro,
        0.0,
        np.where(r < rc, 6 * (rc - r) * (ro - r) / (rc - ro) ** 3, 0.0),
    )
