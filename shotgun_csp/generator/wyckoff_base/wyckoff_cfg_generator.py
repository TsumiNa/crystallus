# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0


from copy import deepcopy
from typing import Dict, Sequence, Union
from xmlrpc.client import Boolean

from shotgun_csp._libcrystal import WyckoffCfgGenerator as _WYG


class WyckoffCfgGenerator(object):
    def __init__(
        self,
        composition,
        *,
        max_recurrent: int = 1_000,
        n_jobs: int = -1,
        priority: Union[Dict[int, Dict[str, float]], None] = None,
        verbose: Boolean = False,
    ):
        """A generator for possible Wyckoff configuration generation.

        Parameters
        ----------
        max_recurrent:
            Max recurrent until generate a reasonable structure, by default 5_000
        n_jobs:
            Number of cpu cores when parallel calculation, by default -1
        priority:
            Priorities for Wyckoff letters. By default, a Wyckoff letter will be sampled
            from an Uniform distribution of the all available letters.
            Give this parameter will overwrite the corresponding priority list of Wyckoff letters.
            For example, space group 167 has Wyckoff letters `[a, b, c, d, e, f]`
            If priority is None, all wyckoff letters will be selected under uniform distribution.
            Now, we want to lift the priority `a`, `b` and `d`, we can give parameter `priority`
            a values like this: `{167: {a: 2, b: 2, d: 2}}`. After that, the
            new priority change to `{a: 2, b: 2, c: 0, d: 2, e: 0, f: 0}`. When generating,
            the priority list will be normalized as this `{a: 2/6, b: 2/6, c: 0/6, d: 2/6 e: 0/6, f: 0/6}`.
        composition:
            Composition of compounds in the primitive cell; should be formatted
            as {<element symbol>: <ratio in float>}.
        """

        self._wyg = _WYG(composition, max_recurrent=max_recurrent, n_jobs=n_jobs, priority=priority, verbose=verbose)
        self._priority = priority
        self._composition = composition

    @property
    def max_recurrent(self):
        """
        Retrieve the maximum recurrent value from the Wyckoff generator.

        Returns:
            int: The maximum recurrent value.
        """
        return self._wyg.max_recurrent

    @property
    def n_jobs(self):
        """
        Returns the number of jobs.

        This method retrieves the number of jobs from the `_wyg` attribute.

        Returns:
            int: The number of jobs.
        """
        return self._wyg.n_jobs

    @n_jobs.setter
    def n_jobs(self, n):
        self._wyg.n_jobs = n

    @property
    def verbose(self):
        """
        Returns the verbosity setting of the Wyckoff generator.

        :return: Verbosity setting of the Wyckoff generator.
        :rtype: bool
        """
        return self._wyg.verbose

    @verbose.setter
    def verbose(self, n):
        self._wyg.verbose = n

    @property
    def composition(self):
        """
        Returns a deep copy of the composition.

        This method provides a deep copy of the `_composition` attribute to ensure
        that the original composition data remains unaltered when modifications are
        made to the returned copy.

        Returns:
            dict: A deep copy of the composition.
        """
        return deepcopy(self._composition)

    @property
    def priority(self):
        """
        Returns a deep copy of the _priority attribute.

        This method ensures that the original _priority attribute is not modified
        by returning a deep copy of it.

        Returns:
            Any: A deep copy of the _priority attribute.
        """
        return deepcopy(self._priority)

    def gen_one(self, *, spacegroup_num: int):
        """Try to generate a possible Wyckoff configuration under the given space group.

        Parameters
        ----------
        spacegroup_num:
            Space group number.

        Returns
        -------
        Dict
            Wyckoff configuration set, which is a dict with format like:
            {"Li": ["a", "c"], "O": ["i"]}. Here, the "Li" is an available element
            symbol and ["a", "c"] is a list which contains coresponding Wyckoff
            letters. For convenience, dict will be sorted by keys.
        """
        return self._wyg.gen_one(spacegroup_num)

    def gen_many(self, size: int, *, spacegroup_num: Union[int, Sequence[int]]):
        """Try to generate possible Wyckoff configuration sets.

        Parameters
        ----------
        size:
            How many times to try for one space group.
        spacegroup_num:
            Spacegroup numbers to generate Wyckoff configurations.

        Returns
        -------
        Dict[int, List[Dict]], List[Dict]
            A collection contains spacegroup number and it's corresponding Wyckoff
            configurations (wy_cfg). If only one spacegroup number was given,
            will only return the list of wy_cfgs, otherwise return in dict with
            spacegroup number as key. wy_cfgs will be formated as
            {element 1: [Wyckoff_letter, Wyckoff_letter, ...], element 2: [...], ...}.
        """
        if isinstance(spacegroup_num, int):
            spacegroup_num = (spacegroup_num,)
        return self._wyg.gen_many(size, *spacegroup_num)

    def gen_many_iter(self, size: int, *, spacegroup_num: Union[int, Sequence[int]]):
        """Try to generate possible Wyckoff configuration sets.

        Parameters
        ----------
        size:
            How many times to try for one space group.
        spacegroup_num:
            Spacegroup numbers to generate Wyckoff configurations.

        Yields
        ------
        Dict[int, List[Dict]], List[Dict]
            A collection contains spacegroup number and it's corresponding Wyckoff
            configurations (wy_cfg). If only one spacegroup number was given,
            will only return the list of wy_cfgs, otherwise return in dict with
            spacegroup number as key. wy_cfgs will be formated as
            {element 1: [Wyckoff_letter, Wyckoff_letter, ...], element 2: [...], ...}.
        """
        if isinstance(spacegroup_num, int):
            spacegroup_num = (spacegroup_num,)
        for sp_num in spacegroup_num:
            yield sp_num, self._wyg.gen_many(size, sp_num)

    def __repr__(self):
        return f"WyckoffCfgGenerator(\
            \n    max_recurrent={self.max_recurrent},\
            \n    n_jobs={self.n_jobs}\
            \n    priority={self._priority}\
            \n    composition={self._composition}\
            \n    verbose={self.verbose}\
            \n)"
