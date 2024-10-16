# Copyright 2024 TsumiNa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple, Union


from ._core import CrystalGenerator as _CG

__all__ = ["CrystalGenerator"]


class CrystalGenerator(object):

    def __init__(self,
                 spacegroup_num: int,
                 volume_of_cell: float,
                 variance_of_volume: float,
                 *,
                 angle_range: Tuple[float, float] = (30., 150.),
                 angle_tolerance: float = 20.,
                 lattice: Union[Tuple[float], None] = None,
                 empirical_coords: Union[List[Tuple[str, List[float]]], None] = None,
                 empirical_coords_variance: float = 0.01,
                 empirical_coords_sampling_rate: float = 1.,
                 empirical_coords_loose_sampling: bool = True,
                 max_attempts_number: int = 5_000,
                 n_jobs: int = -1,
                 verbose: bool = False):
        """A generator for possible crystal structure generation.

        Parameters
        ----------
        spacegroup_num:
            Specify the spacegroup.
        volume_of_cell:
            The estimated volume of primitive cell. Unit is Å^3.
        variance_of_volume:
            The estimated variance of volume prediction. Unit is Å^3.
            ``volume_of_cell`` and ``variance_of_volume`` will be used to build
            a Gaussion distribution for the sampling of volume of primitive cell.
            We will use the abstract valuse if sampled volume is negative.
        angle_range:
            The range of the degree of angles when lattice generation. by default (30., 150.)
        angle_tolerance:
            The Tolerance of minimum of the degree of angles when lattice generation, by default 20.
        lattice:
            Lattice in symmetric cell. optional.
            Given this parameter will specify all angles and the aspect ration between lengths of the lattice.
            The certain lengths will be calculated from volume. Default ``None``.
        empirical_coords:
            Empirical distributuion of atomic coordinates. The coordinates should be given in Wyckoff position
            format. For example: for Wyckoff letter `c` in space group 167, the corresponding positions are
            `(0,0,z) (0,0,-z+1/2) (0,0,-z) (0,0,z+1/2)`. So a fraction coordinate in lattice such as
            `[[0,0,0.3] [0,0,0.2] [0,0,0.7] [0,0,0.8]]` should be converted to `([0, 0, 0.3])`.
            The key of each coordinates pool has two mode, the one is using Wyckoff position letter, for example
            `[('a', [0, 0, 0.3]), ('b': [...])]`, the other is by element and Wyckoff position letter,
            for example `[('Ca:a': [0, 0, 0.3]), ('Ca:b': [...])], ('C:a': [...])]`. You can use parameter
            `empirical_coords_loose_sampling` to switch modes.
        empirical_coords_variance:
            The variance of empirical_coords. This parameter will be used to build a Gaussian distribution.
            The generator will sample values from the distribution as the perturbation of empirical coordinates.
        empirical_coords_sampling_rate:
            The sampling rate when sampling the empirical coordinates.
            A higher rate means instead of random generation,
            sampling from empirical coordinate distribution as more as possible when generating.
            Must be in 0 ~ 1, default 1.
        empirical_coords_loose_sampling:
            Specify the sampling mode. `True` for using Wyckoff position letter, `False` means using
            element and Wyckoff position letter. See parameter `empirical_coords` for details.
        max_attempts_number : int, optional
            Max recurrent until generate a reasonable lattice, by default is 5_000
        n_jobs : int, optional
            Number of cpu cores when parallel calculation, by default -1
        verbose: bool, optional
            Set to ``True`` to show more information.
        """
        # pyo3 can not convert PyList to PyTuple automatically
        if empirical_coords is not None and not isinstance(empirical_coords, tuple):
            empirical_coords = tuple(empirical_coords)
        if lattice is not None:
            lattice = np.asarray(lattice)
            if lattice.size != 9:
                raise ValueError('illegal lattice')
            self._lattice = lattice.reshape(3, 3)
            lattice = tuple(lattice.flatten().tolist())
        else:
            self._lattice = lattice

        self._verbose = verbose
        self._volume_of_cell = volume_of_cell
        self._variance_of_volume = variance_of_volume
        self._angle_range = angle_range
        self._angle_tolerance = angle_tolerance
        self._empirical_coords = empirical_coords
        self._empirical_coords_variance = empirical_coords_variance
        self._empirical_coords_sampling_rate = empirical_coords_sampling_rate
        self._empirical_coords_loose_sampling = empirical_coords_loose_sampling

        self._cg = _CG(spacegroup_num=spacegroup_num,
                       volume_of_cell=volume_of_cell,
                       variance_of_volume=variance_of_volume,
                       angle_range=angle_range,
                       angle_tolerance=angle_tolerance,
                       lattice=lattice,
                       empirical_coords=empirical_coords,
                       empirical_coords_variance=empirical_coords_variance,
                       empirical_coords_sampling_rate=empirical_coords_sampling_rate,
                       empirical_coords_loose_sampling=empirical_coords_loose_sampling,
                       max_attempts_number=max_attempts_number,
                       n_jobs=n_jobs,
                       verbose=verbose)

    @property
    def volume_of_cell(self):
        return self._volume_of_cell

    @property
    def variance_of_volume(self):
        return self._variance_of_volume

    @property
    def angle_range(self):
        return self._angle_range

    @property
    def angle_tolerance(self):
        return self._angle_tolerance

    @property
    def max_attempts_number(self):
        return self._cg.max_attempts_number

    @property
    def empirical_coords(self):
        return deepcopy(self._empirical_coords)

    @property
    def lattice(self):
        return deepcopy(self._lattice)

    @property
    def empirical_coords_variance(self):
        return self._empirical_coords_variance

    @property
    def empirical_coords_sampling_rate(self):
        return self._empirical_coords_sampling_rate

    @property
    def empirical_coords_loose_sampling(self):
        return self._empirical_coords_loose_sampling

    @property
    def spacegroup_num(self):
        return self._cg.spacegroup_num

    @property
    def verbose(self):
        return self._verbose

    @property
    def n_jobs(self):
        return self._cg.n_jobs

    @n_jobs.setter
    def n_jobs(self, n):
        self._cg.n_jobs = n

    def gen_one(
        self,
        wyckoff_cfg: Dict[str, Tuple[str]],
        *,
        check_distance: bool = True,
        distance_scale_factor: float = 0.1,
    ):
        """Try to generate a legal crystal structure with given configuration set.

        Parameters
        ----------
        wyckoff_cfg:
            Wyckoff Configuration set, which is a dict with format like:
            {"Li": ["a", "c"], "O": ["i"]}. Here, the "Li" is an available element
            symbol and ["a", "c"] is a list which contains coresponding Wyckoff
            letters. For convenience, dict will be sorted by keys.
        check_distance:
            Whether the atomic distance should be checked. default ``True``
        distance_scale_factor:
            Scale factor to determine the tolerance of atomic distances when distance checking. Unit is Å,
            When ``check_distance`` is ``True``, Any structure has
            all_atomic_distance < (A_atom_covalent_radius + B_atom_covalent_radius) * (1 - distance_scale_factor) will be rejected,
            by default 0.1

        Returns
        -------
        Dict
            Structure information contains ``spacegroup_mun: int``,
            ``volume: float``, ``lattice: list``, ``wyckoff_letters: list``,
            and ``coords: list``.
        """
        return self._cg.gen_one(wyckoff_cfg, check_distance=check_distance, distance_scale_factor=distance_scale_factor)

    def gen_many(
        self,
        expect_size: int,
        *wyckoff_cfgs: Dict[str, Tuple[str]],
        max_attempts: Union[int, None] = None,
        check_distance: bool = True,
        distance_scale_factor: float = 0.1,
    ) -> List[Dict]:
        """Try to generate legal crystal structures with given configuration set(s).

        Parameters
        ----------
        expect_size:
            The expectation of the total amount of generated structures based on one Wyckoff.
            Whatever one generated structure is legal or not, **one attempt** will be consumed. 
            Please noted that the result could be empty when no structures matched the atomic distance conditions.
            When the number of generated structures are not fit your expectation too far away,
            try to give the parameter ``max_attempts`` a higher value..
        *wyckoff_cfgs:
            A tuple with Wyckoff configuration set(s).
            Wyckoff Configuration set is a dict with format like: {"Li": ["a", "c"], "O": ["i"]}.
            Here, the "Li" is an available element symbol and ["a", "c"] is a list
            which contains coresponding Wyckoff letters. For convenience, dict will
            be sorted by keys.
        max_attempts:
            Specify the max number of attempts in structure generation.
            When the number of generated structures is small than ``expect_size``, new rounds of structure generation will be performed.
            The generation will stop until the number of generated structures is more than ``expect_size, `` or the total attempts reach the ``max_attempts``.
            Default ``None``, means ``max_attempts`` equal to parameter ``expect_size``.
        check_distance:
            Whether the atomic distance should be checked. default ``True``
        distance_scale_factor:
            Scale factor to determine the tolerance of atomic distances when distance checking. Unit is Å,
            When ``check_distance`` is ``True``, Any structure has
            all_atomic_distance < (A_atom_covalent_radius + B_atom_covalent_radius) * (1 - distance_scale_factor) will be rejected,
            by default 0.1

        Returns
        -------
        Dict
            Structure information contains ``spacegroup_mun: int``,
            ``volume: float``, ``lattice: list``, ``wyckoff_letters: list``,
            and ``coords: list``.
        """
        assert expect_size >= 1, 'attempts number must be greater than 1'

        if len(wyckoff_cfgs) > 0:
            return self._cg.gen_many(
                expect_size,
                wyckoff_cfgs,
                max_attempts=max_attempts,
                check_distance=check_distance,
                distance_scale_factor=distance_scale_factor,
            )
        return []

    def gen_many_iter(
        self,
        expect_size: int,
        *wyckoff_cfgs: Dict[str, Tuple[str]],
        max_attempts: Union[int, None] = None,
        check_distance: bool = True,
        distance_scale_factor: float = 0.1,
    ):
        """Try to generate legal crystal structures with given configuration set(s), iteratively.

        Parameters
        ----------
        expect_size: int
            The expectation of the total amount of generated structures based on one Wyckoff.
            Whatever one generated structure is legal or not, **one attempt** will be consumed. 
            Please noted that the result could be empty when no structures matched the atomic distance conditions.
            When the number of generated structures are not fit your expectation too far away,
            try to give the parameter ``max_attempts`` a higher value..
        max_attempts: Union[int, None], optional
            Specify the max number of attempts in structure generation.
            When the number of generated structures is small than ``expect_size``, new rounds of structure generation will be performed.
            The generation will stop until the number of generated structures is more than ``expect_size, `` or the total attempts reach the ``max_attempts``.
            Default ``None``, means ``max_attempts`` equal to parameter ``expect_size``.
        check_distance: bool, optional
            Whether the atomic distance should be checked. default ``True``
        distance_scale_factor : float, optional
            Scale factor to determine the tolerance of atomic distances when distance checking. Unit is Å,
            When ``check_distance`` is ``True``, Any structure has
            all_atomic_distance < (A_atom_covalent_radius + B_atom_covalent_radius) * (1 - distance_scale_factor) will be rejected,
            by default 0.1
        *wyckoff_cfgs: Dict[str, Tuple[str]]
            A tuple with Wyckoff configuration set(s).
            Wyckoff Configuration set is a dict with format like: {"Li": ["a", "c"], "O": ["i"]}.
            Here, the "Li" is an available element symbol and ["a", "c"] is a list
            which contains coresponding Wyckoff letters. For convenience, dict will
            be sorted by keys..
        Yields
        ------
        Tuple[Dict]
            Structure information contains ``spacegroup_mun: int``,
            ``volume: float``, ``lattice: list``, ``wyckoff_letters: list``,
            and ``coords: list``.
        """
        assert expect_size >= 1, 'attempts number must be greater than 1'
        for cfg in wyckoff_cfgs:
            yield cfg, self._cg.gen_many(
                expect_size,
                (cfg,),
                max_attempts=max_attempts,
                check_distance=check_distance,
                distance_scale_factor=distance_scale_factor,
            )

    def __repr__(self):
        return f"CrystalGenerator(\
            \n    spacegroup_num={self.spacegroup_num},\
            \n    volume_of_cell={self.volume_of_cell},\
            \n    variance_of_volume={self.variance_of_volume},\
            \n    angle_range={self.angle_range},\
            \n    angle_tolerance={self.angle_tolerance},\
            \n    max_attempts_number={self.max_attempts_number},\
            \n    lattice={'...' if self._lattice is not None else None},\
            \n    empirical_coords={'...' if self._empirical_coords is not None else None},\
            \n    empirical_coords_variance={self.empirical_coords_variance},\
            \n    empirical_coords_sampling_rate={self.empirical_coords_sampling_rate},\
            \n    empirical_coords_loose_sampling={self.empirical_coords_loose_sampling},\
            \n    verbose={self.verbose}\
            \n    n_jobs={self.n_jobs}\
            \n)"
