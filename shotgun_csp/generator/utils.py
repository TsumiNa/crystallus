# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import gc
import re
import time
import warnings
from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixSymmetry
from ase.optimize import FIRE
from joblib import Parallel, delayed
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from shotgun_csp.descriptor import Compositions
from shotgun_csp.model.cgcnn import CrystalGraphConvNet
from shotgun_csp.model.extension import TensorConverter
from shotgun_csp.model.sequential import SequentialLinear
from shotgun_csp.model.training import Checker, Trainer
from shotgun_csp.utils import SpaceGroupDB

__call__ = [
    "WyckoffPositionConverter",
    "get_equivalent_coords",
    "lll_reduce",
    "pbc_all_distances",
    "structure_reconstructor",
    "predict_volume",
    "calculate_dissimilarity",
    "convert_struct_to_primitive_with_volume",
]


def predict_volume(
    compositions: list[Union[str, Composition, dict]],
    pred_model: Union[str, Path],
    *,
    checkpoint="mae",
    ignore_warn=True,
    n_jobs=-1,
) -> np.array:
    """
    Predict the volume of the structures.

    Args:
        compositions: The compositions.
        pred_model: The model for prediction.
        checkpoint: The checkpoint for the prediction. Default is "mae".
        ignore_warn: Whether to ignore the warnings. Default is True.
    """
    if ignore_warn:
        warnings.filterwarnings("ignore")

    torch.serialization.add_safe_globals([SequentialLinear, CrystalGraphConvNet])
    checker = Checker(pred_model)
    trainer = Trainer.from_checker(checker=checker).extend(TensorConverter())
    desc = Compositions(n_jobs=n_jobs).transform(compositions)

    if checkpoint is None:
        return trainer.predict(x_in=desc).flatten()
    return trainer.predict(x_in=desc, checkpoint=checkpoint).flatten()


def calculate_dissimilarity(anchor_structure: Structure, *structure: Structure, n_jobs=10):
    """Calculate the dissimilarity between the anchor structure and the other structures.

    Args:
        anchor_structure: The anchor structure.
        structure: The other structures.
        n_jobs: The number of jobs to run in parallel. Default is 10.

    Returns:
        np.array: The dissimilarity between the anchor structure and the other structures.
    """
    ssf = SiteStatsFingerprint(
        CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=0),
        stats=("mean", "std_dev", "minimum", "maximum"),
    )
    v_anchor = np.array(ssf.featurize(anchor_structure))
    tmp = Parallel(n_jobs=n_jobs)(delayed(ssf.featurize)(s) for s in structure)
    return [np.linalg.norm(np.array(s) - v_anchor) for s in tmp]


def convert_struct_to_primitive_with_volume(structure: Structure, volume: Union[None, float] = None) -> Structure:
    """
    Convert the structure to a primitive structure and adjust the volume of the unit cell.

    Args:
        structure (Structure): The input structure.
        volume (float): The volume of the unit cell. Default is None.

    Returns:
        Structure: The modified structure.
    """
    structure = structure.get_primitive_structure()
    if volume is not None:
        structure.scale_lattice(volume)
    return structure


class WyckoffPositionConverter:
    """Convert fraction coordinates into Wyckoff position formate."""

    class _Coordinate:
        patten = re.compile(r"(?P<xyz>-?\d?[xyz])|(?P<cons_frac>\d\/\d?)|(?P<cons>\d)")

        @classmethod
        def _inner(cls, s):
            if "-" in s:
                coeff = -1
            else:
                coeff = 1
            if "2" in s:
                coeff *= 2

            return coeff

        def __call__(self, coordinates):
            const = 0
            x_coeff, y_coeff, z_coeff, const = 0, 0, 0, 0

            for e in self.patten.findall(coordinates):
                if e[0] != "":
                    s = e[0].lower()
                    if "x" in s:
                        x_coeff = self._inner(s)
                        continue

                    if "y" in s:
                        y_coeff = self._inner(s)
                        continue

                    if "z" in s:
                        z_coeff = self._inner(s)
                        continue

                if e[1] != "":
                    s = e[1].split("/")
                    const = float(s[0]) / float(s[1])
                    continue

                if e[2] != "":
                    const = float(e[2])
                    continue

            return x_coeff, y_coeff, z_coeff, const

    class _Particle:
        patten = re.compile(r",\s*")

        def __init__(self):
            self.Coordinate = WyckoffPositionConverter._Coordinate()

        def __call__(self, position):
            return [self.Coordinate(coord) for coord in self.patten.split(position)]

    patten = re.compile(r"(?<=\)),\s*")

    def __init__(self, spacegroup_num: int):
        wys = SpaceGroupDB.get(SpaceGroupDB.spacegroup_num == spacegroup_num).wyckoffs
        self.particle = WyckoffPositionConverter._Particle()
        self.wyckoff_pos = {wy.letter: self.patten.split(wy.positions)[0][1:-1] for wy in wys}

    def _inner(self, wy_letter, coord):
        b = np.asarray(coord)
        a = np.array(self.particle(self.wyckoff_pos[wy_letter]))
        idx = []

        if np.count_nonzero(a[:, 0]):
            idx.append(0)
        if np.count_nonzero(a[:, 1]):
            idx.append(1)
        if np.count_nonzero(a[:, 2]):
            idx.append(2)
        b[idx] -= a[idx, -1]

        if len(idx) > 1:
            solves = np.linalg.solve(a[idx][:, idx], b[idx] - a[idx, -1])
            b[idx] = solves

        return b.tolist()

    def __call__(
        self,
        wyckoff_letters: Union[str, pd.Series, List[str]],
        coords: Union[str, pd.Series, List[Tuple[float, float, float]]],
        elements: Union[str, pd.Series, List[str], None] = None,
        *,
        data: pd.DataFrame = None,
    ):
        if data is not None:
            if not isinstance(wyckoff_letters, str) or not isinstance(coords, str):
                raise ValueError("`wyckoff_letters` and `coords` must be the column name when `data` is set")

            if elements is not None and isinstance(elements, str):
                wy_and_coord = [a for _, a in data[[wyckoff_letters, coords, elements]].iterrows()]
            else:
                wy_and_coord = [a for _, a in data[[wyckoff_letters, coords]].iterrows()]
        else:
            if isinstance(wyckoff_letters, str) or isinstance(coords, str) or isinstance(elements, str):
                raise ValueError(
                    "found `wyckoff_letters`, `coords`, and `elements` as column name but `data` is not given"
                )

            if elements is None:
                if not len(wyckoff_letters) == len(coords):
                    raise ValueError("`wyckoff_letters`and `coords` have different lengths")
                wy_and_coord = list(zip(wyckoff_letters, coords))
            else:
                if not len(wyckoff_letters) == len(coords) == len(elements):
                    raise ValueError("`wyckoff_letters`, `coords`, and `elements` have different lengths")
                wy_and_coord = list(zip(wyckoff_letters, coords, elements))

        if len(wy_and_coord[0]) == 2:
            return [(wy, self._inner(wy, b)) for wy, b in wy_and_coord]
        if len(wy_and_coord[0]) == 3:
            return [(f"{elem}:{wy}", self._inner(wy, b)) for wy, b, elem in wy_and_coord]
        raise ValueError("`wy_and_coord` must be a list of (wyckoff_letter, coord) or (wyckoff_letter, coord, element)")


def structure_reconstructor(
    structure_list,
    *,
    return_primitive=True,
    return_structure_as_dict=True,
    relax_with_ase: Calculator = None,
    scaling_factor=3,
    n_jobs=-1,
    backend="loky",
):
    """
    Reconstructs structures from a list of structure dictionaries and optionally relaxes them using ASE.

    Parameters:
    -----------
    structure_list : list
        List of structure dictionaries, each containing 'lattice', 'species', 'coords', and 'wyckoff_letters'.
    return_primitive : bool, optional
        If True, return the primitive standard structure. If False, return the conventional standard structure.
        Default is True.
    return_structure_as_dict : bool, optional
        If True, return the structure as a dictionary. If False, return the structure as a pymatgen Structure object.
        Default is True.
    relax_with_ase : Calculator, optional
        ASE calculator to use for relaxing the structure. If None, no relaxation is performed. Default is None.
    scaling_factor : int, optional
        Factor by which to scale the lattice volume before relaxation. Default is 3.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is -1 (use all available processors).
    backend : str, optional
        Backend to use for parallel processing. Default is "loky".

    Returns:
    --------
    pd.DataFrame
        DataFrame containing reconstructed structure information, including volume, formula, composition,
        reduced formula, number of atoms, space group, species, structure, elapsed time (if relaxation is performed),
        minimum and maximum interatomic distances, Wyckoff letters, and any errors encountered.
    """

    gc.disable()

    adaptor = AseAtomsAdaptor()

    def _inner(struct_):
        tmp = {}
        try:
            struct = Structure(lattice=struct_["lattice"], species=struct_["species"], coords=struct_["coords"])
            composition = struct.composition
            struct.scale_lattice(struct.volume * scaling_factor)
            atoms = adaptor.get_atoms(struct)
            elapsed_time = None

            if relax_with_ase:
                atoms.set_calculator(relax_with_ase)
                atoms.set_constraint(
                    [
                        FixSymmetry(atoms, symprec=1e-5, adjust_positions=True, adjust_cell=False, verbose=False),
                        FixAtoms(indices=[0]),
                    ]
                )

                # optimaize
                # Start timer
                start_time = time.time()
                opt = FIRE(
                    atoms,
                    # logfile=f"{str(composition).replace(' ', '')}_relax.log",
                    logfile=None,
                    trajectory=None,
                )
                opt.run(fmax=0.1, steps=300)
                atoms.set_constraint()

                # End timer
                end_time = time.time()

                # Calculate elapsed time
                elapsed_time = end_time - start_time

                # return to pymatgen structures
                struct = adaptor.get_structure(atoms)
                struct = Structure.from_dict(struct.as_dict())

            spg_analyer = SpacegroupAnalyzer(struct, symprec=1e-4)
            struct = (
                spg_analyer.get_primitive_standard_structure()
                if return_primitive
                else spg_analyer.get_conventional_standard_structure()
            )
            struct.scale_lattice(struct.volume * 1 / scaling_factor)

            # space group info
            spg_num = spg_analyer.get_space_group_number()
            ratio_ = tuple(sorted([v for v in composition.as_dict().values()]))
            ratioSG_ = f"{'_'.join([str(s) for s in ratio_])}-{spg_num}"

            # symm = SymmetryAnalyzer(atoms, symmetry_tol=1e-4)
            # tmp_ = [(g.wyckoff_letter, g.element, g.multiplicity) for g in symm.get_wyckoff_sets_conventional()]
            # tmp_ = pd.DataFrame(tmp_, columns=("wy_letter", "element", "multiplicity"))
            # c = Counter(tmp_["wy_letter"].values)
            # hist = {k: v / len(tmp_) for k, v in c.items()}

            # # wyckoff
            # d = defaultdict(list)
            # for _, data in tmp_.iterrows():
            #     d[data.element].append(data.wy_letter)
            # wy = {k: sorted(v) for k, v in d.items()}

            # SGwy_ = [f"{g.wyckoff_letter}-{g.multiplicity}" for g in symm.get_wyckoff_sets_conventional()]
            # SGwy_.insert(0, str(spg_num))
            # SGwy_ = "_".join(SGwy_)

            # basic info
            tmp["volume"] = struct.volume
            tmp["formula"] = composition.formula
            tmp["composition"] = dict(composition.as_dict())
            tmp["reduced_formula"] = composition.reduced_formula
            tmp["num_atoms"] = composition.num_atoms
            tmp["space_group"] = spg_analyer.get_space_group_symbol()
            tmp["space_group_num"] = spg_num
            tmp["species"] = struct_["species"]

            # structure info
            tmp["structure"] = struct.as_dict() if return_structure_as_dict else struct
            if elapsed_time:
                tmp["elapsed_time"] = elapsed_time
            dist_ = struct.distance_matrix
            max_ = dist_.max()
            np.fill_diagonal(dist_, np.inf)
            min_ = dist_.min()
            tmp["min_max_dist"] = (max_, min_)
            tmp["wy_letters"] = struct_["wyckoff_letters"]
            # tmp["wy_pattern"] = wy
            # tmp["wy_hist"] = hist
            # tmp["SGwy"] = SGwy_
            tmp["ratio"] = ratio_
            tmp["ratioSG"] = ratioSG_

            # error
            tmp["errors_msg"] = None
        except Exception as e:
            tmp["errors_msg"] = f"{e}"
            # print(e)
            # raise e

        return tmp

    structure_cans = Parallel(n_jobs=n_jobs, backend=backend)(delayed(_inner)(s) for s in structure_list)

    gc.enable()
    gc.collect()

    return pd.DataFrame(structure_cans)


def get_equivalent_coords(structure: Structure, *, mapper: Callable[[str, str, int], str] = None):
    """Extract the equivalent coordinates from the given structure.

    Parameters
    ----------
    structure:
        A pymatgen structure object.
    mapper:
        Specify how to replace the elements. optional.
        ``mapper`` has signature ``[element, wyckoff_letter, multiplicity] -> target_element``
        If this parameter is given, will map the element in the structure to the corresponding one.
        For example, replace the `Ca` in `CaCO2` with `Mg`.

    Returns
    -------
    DataFrame
        A dataframe contains all equivalent coordinates and their Wyckoff position letters.
    """
    struct = SpacegroupAnalyzer(structure).get_symmetrized_structure()

    def _inner(i, sites, mapper=None):
        site = sites[0]
        wy_symbol = struct.wyckoff_symbols[i]
        row = {"element": site.species_string}
        row["spacegroup_num"] = struct.get_space_group_info()[1]
        row["multiplicity"] = int(wy_symbol[:-1])
        row["wyckoff_letter"] = wy_symbol[-1]
        if mapper is not None:
            row["target_element"] = mapper(site.species_string, row["wyckoff_letter"], row["multiplicity"])
        row["coordinate"] = list(site.frac_coords)

        return row

    return pd.DataFrame([_inner(i, sites, mapper=mapper) for i, sites in enumerate(struct.equivalent_sites)])
