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

from shotgun_csp.generator.wyckoff_base import WyckoffCfgGenerator


def test_wyckoff_gen_init():
    """
    Test the initialization of the WyckoffCfgGenerator class.

    This test verifies that the WyckoffCfgGenerator is correctly initialized with the given composition and priority.

    Test cases:
    1. Initialize WyckoffCfgGenerator with a composition dictionary and verify the composition and priority attributes.
    2. Initialize WyckoffCfgGenerator with a composition dictionary and a priority dictionary, then verify the composition and priority attributes.

    Assertions:
    - The composition attribute should match the input composition dictionary.
    - The priority attribute should be None if not provided.
    - The priority attribute should match the input priority dictionary if provided.
    """
    wy = WyckoffCfgGenerator(dict(Ca=2, C=2, O=6))
    assert wy.composition == dict(Ca=2, C=2, O=6)
    assert wy.priority is None

    wy = WyckoffCfgGenerator(dict(Ca=2, C=2, O=6), priority={167: {"e": 0}})
    assert wy.composition == dict(Ca=2, C=2, O=6)
    assert wy.priority == {167: {"e": 0}}


def test_wyckoff_gen_without_priority():
    """
    Test the WyckoffCfgGenerator without priority settings.

    This test performs the following checks:
    1. Generates a single configuration for spacegroup number 167 and asserts
       that it matches one of the expected configurations.
    2. Generates 1000 configurations for spacegroup number 167 and asserts
       that all expected configurations are present in the generated list.
    3. Generates 1000 configurations for spacegroup numbers 167 and 166,
       asserts that the result is a dictionary with keys for each spacegroup
       number, and verifies the presence of both spacegroup numbers in the
       dictionary.
    """
    wy = WyckoffCfgGenerator(dict(Ca=2, C=2, O=6), verbose=False)

    cfg = wy.gen_one(spacegroup_num=167)
    assert cfg in [
        {"Ca": ["b"], "C": ["a"], "O": ["e"]},
        {"Ca": ["b"], "C": ["a"], "O": ["d"]},
        {"Ca": ["a"], "C": ["b"], "O": ["e"]},
        {"Ca": ["a"], "C": ["b"], "O": ["d"]},
    ]

    cfgs = wy.gen_many(1000, spacegroup_num=167)
    assert all(
        [
            cfg in cfgs
            for cfg in [
                {"Ca": ["b"], "C": ["a"], "O": ["e"]},
                {"Ca": ["b"], "C": ["a"], "O": ["d"]},
                {"Ca": ["a"], "C": ["b"], "O": ["e"]},
                {"Ca": ["a"], "C": ["b"], "O": ["d"]},
            ]
        ]
    )

    cfgs = wy.gen_many(1000, spacegroup_num=(167, 166))
    assert isinstance(cfgs, dict)
    assert len(cfgs) == 2
    assert 166 in cfgs
    assert 167 in cfgs


def test_wyckoff_gen_with_priority():
    """
    Test the WyckoffCfgGenerator with a specified priority for Wyckoff positions.

    This test checks the following:
    1. Generates a single configuration for space group 167 and verifies it matches one of the expected configurations.
    2. Generates multiple configurations (1000) for space group 167 and verifies that all expected configurations are present in the generated configurations.

    The WyckoffCfgGenerator is initialized with a composition of Ca, C, and O atoms and a priority dictionary for Wyckoff positions in space group 167.
    """
    wy = WyckoffCfgGenerator(dict(Ca=2, C=2, O=6), priority={167: {"e": 0, "d": 2, "b": 2, "a": 2, "c": 2, "f": 2}})
    cfg = wy.gen_one(spacegroup_num=167)
    assert cfg in [{"Ca": ["b"], "C": ["a"], "O": ["d"]}, {"Ca": ["a"], "C": ["b"], "O": ["d"]}]
    cfgs = wy.gen_many(1000, spacegroup_num=167)
    assert all([cfg in cfgs for cfg in [{"Ca": ["b"], "C": ["a"], "O": ["d"]}, {"Ca": ["a"], "C": ["b"], "O": ["d"]}]])
