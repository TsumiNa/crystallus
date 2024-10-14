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
import numpy.testing as npt
import pytest
from pymatgen.core import Lattice

from shotgun_csp.generator.wyckoff_base import CrystalGenerator


def test_crystal_gen_one():
    """
    Test the `gen_one` method of the `CrystalGenerator` class.

    This test verifies that the `gen_one` method generates a crystal structure
    with the expected properties when provided with specific input parameters.

    Assertions:
    - The generated structure contains the expected keys: "lattice", "volume",
        "spacegroup_num", "species", "wyckoff_letters", and "coords".
    - The volume of the generated structure is within the range [940, 1060].
    - The lattice parameter is consistent with the cubic root of the volume.
    - The space group number of the generated structure is 207.
    - The species in the generated structure are ['C', 'C', 'O', 'O', 'O'].
    - The Wyckoff letters in the generated structure are ['a', 'b', 'd', 'd', 'd'].
    - The coordinates of the atoms in the generated structure are as expected.
    """
    cg = CrystalGenerator(207, 1000, 20)
    structure = cg.gen_one(dict(C=("a", "b"), O=("d",)))

    assert set(structure.keys()) == {
        "lattice",
        "volume",
        "spacegroup_num",
        "species",
        "wyckoff_letters",
        "coords",
    }
    assert 940 <= structure["volume"] <= 1060
    l = pow(structure["volume"], 1 / 3)
    assert np.allclose(structure["lattice"], [[l, 0, 0], [0, l, 0], [0, 0, l]])
    assert structure["spacegroup_num"] == 207
    assert structure["species"] == ["C", "C", "O", "O", "O"]
    assert structure["wyckoff_letters"] == ["a", "b", "d", "d", "d"]
    assert structure["coords"] == [
        [0.0, 0.0, 0.0],
        [1 / 2, 1 / 2, 1 / 2],
        [1 / 2, 0.0, 0.0],
        [0.0, 1 / 2, 0.0],
        [0.0, 0.0, 1 / 2],
    ]


def test_crystal_gen_one_with_lattice():
    """
    Test the CrystalGenerator's ability to generate a crystal structure with a given lattice.

    This test verifies that the generated crystal structure has the expected properties:
    - The structure contains the correct keys: 'lattice', 'volume', 'spacegroup_num', 'species', 'wyckoff_letters', and 'coords'.
    - The volume of the generated structure is approximately 1000.
    - The ratios of the lattice parameters (abc) of the new structure to the old structure are consistent.
    - The angles of the new lattice are approximately equal to the angles of the old lattice.
    - The angles of the new lattice are approximately 90 degrees.

    The test uses the following steps:
    1. Create a Lattice object with predefined parameters.
    2. Initialize a CrystalGenerator with the given lattice.
    3. Generate a crystal structure using the CrystalGenerator.
    4. Assert that the generated structure contains the expected keys.
    5. Compare the lattice parameters and angles of the generated structure to the original lattice.
    6. Assert that the volume of the generated structure is approximately 1000.
    7. Assert that the lattice angles are approximately 90 degrees.
    """
    lattice = Lattice(
        [
            [1.53132450e01, 0.00000000e00, 9.37665824e-16],
            [-4.66967683e-16, 7.62616100e00, 4.66967683e-16],
            [0.00000000e00, 0.00000000e00, 1.07431550e01],
        ]
    )
    cg = CrystalGenerator(207, 1000, 0, lattice=lattice.matrix)
    structure = cg.gen_one(dict(C=("a", "b"), O=("d",)))

    assert set(structure.keys()) == {
        "lattice",
        "volume",
        "spacegroup_num",
        "species",
        "wyckoff_letters",
        "coords",
    }
    lattice_new = Lattice(structure["lattice"])
    abc_new, angles_new = lattice_new.abc, lattice_new.angles
    abc_old, angles_old = lattice.abc, lattice.angles

    npt.assert_almost_equal(structure["volume"], 1000)
    npt.assert_almost_equal(abc_new[0] / abc_old[0], abc_new[1] / abc_old[1], decimal=6)
    npt.assert_almost_equal(angles_new[0], angles_old[0], decimal=5)
    npt.assert_almost_equal(angles_new[1], angles_old[1], decimal=5)
    npt.assert_almost_equal(angles_new[2], angles_old[2], decimal=5)
    npt.assert_almost_equal(angles_new[0], 90.0, decimal=5)


def test_crystal_gen_one_with_template():
    """
    space group 167
    =========================================================================
    Multiplicity | Wyckoff letter |      Coordinates
    -------------------------------------------------------------------------
            6       |         e      | (x,0,1/4) (0,x,1/4) (-x,-x,1/4)
                    |                | (-x,0,3/4) (0,-x,3/4) (x,x,3/4)
    -------------------------------------------------------------------------
            6       |         d      | (1/2,0,0) (0,1/2,0) (1/2,1/2,0)
                    |                | (0,1/2,1/2) (1/2,0,1/2) (1/2,1/2,1/2)
    -------------------------------------------------------------------------
            4       |         c      | (0,0,z) (0,0,-z+1/2) (0,0,-z) (0,0,z+1/2)
    -------------------------------------------------------------------------
            2       |         b      | (0,0,0) (0,0,1/2)
    -------------------------------------------------------------------------
            2       |         a      | (0,0,1/4) (0,0,3/4)
    -------------------------------------------------------------------------
    """
    template = [("c", [0.2, 0.0, 0.0]), ("e", [0.4, 0.0, 0.0])]
    cg = CrystalGenerator(167, 1000, 10, angle_range=(40, 50), empirical_coords=template, empirical_coords_variance=0)
    structure = cg.gen_one(dict(Li=("c",), P=("e",)))

    assert structure["wyckoff_letters"] == ["c"] * 4 + ["e"] * 6
    npt.assert_almost_equal(
        structure["coords"],
        [
            [0.2, 0.2, 0.2],  # Li
            [0.3, 0.3, 0.3],  # Li
            [0.8, 0.8, 0.8],  # Li
            [0.7, 0.7, 0.7],  # Li
            [0.4, 0.1, 0.25],  # P
            [0.25, 0.4, 0.1],  # P
            [0.1, 0.25, 0.4],  # P
            [0.6, 0.9, 0.75],  # P
            [0.75, 0.6, 0.9],  # P
            [0.9, 0.75, 0.6],  # P
        ],
    )


def test_crystal_gen_many_1():
    """
    Test the `gen_many` method of the `CrystalGenerator` class.

    This test initializes a `CrystalGenerator` instance with specific parameters
    and generates multiple crystal structures. It then asserts that the length
    of the generated structure list is zero.

    Test case:
    - Initialize `CrystalGenerator` with parameters (207, 1000, 20).
    - Generate 10 crystal structures using the `gen_many` method.
    - Assert that the length of the generated structure list is zero.
    """
    cg = CrystalGenerator(207, 1000, 20)
    structure = cg.gen_many(10)

    assert len(structure) == 0


def test_gen_many_2():
    """
    Test the `gen_many` method of `CrystalGenerator` with invalid `max_attempts`.

    This test verifies that the `gen_many` method raises a `ValueError` when the
    `max_attempts` parameter is smaller than the `expect_size` parameter.

    Steps:
    1. Create an instance of `CrystalGenerator` with specific parameters.
    2. Call the `gen_many` method with a `max_attempts` value smaller than `expect_size`.
    3. Assert that a `ValueError` is raised with the appropriate error message.

    Expected Result:
    A `ValueError` is raised with the message "`max_attempts` can not be smaller than `expect_size`".
    """
    cg = CrystalGenerator(207, 1000, 20)
    with pytest.raises(ValueError, match="`max_attempts` can not be smaller than `expect_size`"):
        cg.gen_many(10, {"C": ("a", "b"), "O": ("d",)}, max_attempts=2)


def test_crystal_gen_many_3():
    """
    Test the CrystalGenerator's `gen_many` method with specific parameters.

    This test initializes a CrystalGenerator with space group number 207,
    1000 atoms, and 20 types of atoms. It then generates 10 crystal structures
    with specified species and Wyckoff positions.

    The test asserts the following:
    - The number of generated structures is 10.
    - The Wyckoff letters, species, space group number, and coordinates of
      the first and third generated structures are identical.
    """
    cg = CrystalGenerator(207, 1000, 20)
    structure = cg.gen_many(10, {"C": ("a", "b"), "O": ("d",)})

    assert len(structure) == 10
    assert structure[0]["wyckoff_letters"] == structure[2]["wyckoff_letters"]
    assert structure[0]["species"] == structure[2]["species"]
    assert structure[0]["spacegroup_num"] == structure[2]["spacegroup_num"]
    assert structure[0]["coords"] == structure[2]["coords"]


def test_crystal_gen_many_4():
    """
    Test the `gen_many` method of the `CrystalGenerator` class with specific configurations.

    This test initializes a `CrystalGenerator` instance with specific parameters and generates multiple crystal structures
    using the `gen_many` method. It then verifies the following:
    - The total number of generated structures is as expected.
    - The Wyckoff letters of the first and sixth structures match the expected values.
    - The Wyckoff letters of the first structure match those of the fifth structure.
    - The Wyckoff letters of the sixth structure match those of the tenth structure.

    Assertions:
    - The length of the generated structure list is 10.
    - The Wyckoff letters of the first structure are ['a', 'b', 'd', 'd', 'd'].
    - The Wyckoff letters of the sixth structure are ['b', 'a', 'd', 'd', 'd'].
    - The Wyckoff letters of the first structure match those of the fifth structure.
    - The Wyckoff letters of the sixth structure match those of the tenth structure.
    """
    cg = CrystalGenerator(207, 1000, 20)
    cfgs = (
        {"C": ("a", "b"), "O": ("d",)},
        {"O": ("d",), "C": ("b", "a")},
    )
    structure = cg.gen_many(5, *cfgs)

    assert len(structure) == 10
    assert structure[0]["wyckoff_letters"] == ["a", "b", "d", "d", "d"]
    assert structure[5]["wyckoff_letters"] == ["b", "a", "d", "d", "d"]
    assert structure[0]["wyckoff_letters"] == structure[4]["wyckoff_letters"]
    assert structure[5]["wyckoff_letters"] == structure[9]["wyckoff_letters"]


def test_crystal_gen_many_5():
    """
    Test the `gen_many` method of the `CrystalGenerator` class with various parameters.

    This test performs the following checks:
    1. Generates 100 structures with a given composition and checks that the length of the resulting structure list is 0, indicating a distance error.
    2. Generates 10 structures with the same composition but without distance checking, and verifies that the length of the resulting structure list is 10.
    3. Generates 1000 structures with a relaxed distance condition (distance_scale_factor=0.5) and ensures that the length of the resulting structure list is greater than 0.

    Assertions:
    - The length of the structure list should be 0 when distance checking is enabled.
    - The length of the structure list should be 10 when distance checking is disabled.
    - The length of the structure list should be greater than 0 when the distance condition is relaxed.
    """
    cg = CrystalGenerator(33, 1168, 15)
    comp = {"Ag": ["a", "a", "a", "a", "a", "a", "a", "a"], "Ge": ["a"], "S": ["a", "a", "a", "a", "a", "a"]}

    # distance error
    structure = cg.gen_many(100, comp)
    assert len(structure) == 0

    # no check
    structure = cg.gen_many(10, comp, check_distance=False)
    assert len(structure) == 10

    # make condition losser
    structure = cg.gen_many(1000, comp, distance_scale_factor=0.5)
    assert len(structure) > 0
