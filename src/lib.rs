// Copyright 2024 TsumiNa.
// SPDX-License-Identifier: Apache-2.0

use pyo3::{prelude::*, py_run};

mod crystal_gen;
mod particle_gen;
mod utils;
mod wyckoff_cfg_gen;

use crate::crystal_gen::CrystalGenerator;
use crate::particle_gen::ParticleGenerator;
use crate::wyckoff_cfg_gen::WyckoffCfgGenerator;

#[pymodule]
#[pyo3(name = "_libcrystal")]
fn _core_mod<'py>(py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    // register classes
    m.add_class::<ParticleGenerator>()?;
    m.add_class::<CrystalGenerator>()?;
    m.add_class::<WyckoffCfgGenerator>()?;

    // register functions
    let utils_mod = PyModule::new_bound(m.py(), "_libcrystal.utils")?;
    utils::register(&utils_mod)?;

    // Note that this does not define a package, so this won’t allow Python code to
    // directly import submodules by using from parent_module import child_module.
    // For more information, see https://pyo3.rs/latest/module.html#python-submodules
    py_run!(
        py,
        utils_mod,
        "import sys; sys.modules['shotgun_csp._libcrystal.utils'] = utils_mod"
    );

    // add submodules
    m.add_submodule(&utils_mod)?;
    Ok(())
}
