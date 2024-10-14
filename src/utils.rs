// Copyright 2024 TsumiNa.
// SPDX-License-Identifier: Apache-2.0

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PySequence;

use libcrystal::{
    utils::{lll_reduce as _lll, pbc_all_distances as _pbc},
    Float,
};

// register functions
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lll_reduce_py, m)?)?;
    m.add_function(wrap_pyfunction!(pbc_all_distances_py, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(name = "lll_reduce")]
#[pyo3(signature = (basis, delta=0.75))]
fn lll_reduce_py(
    basis: &Bound<'_, PySequence>,
    delta: Float,
) -> PyResult<(Vec<Float>, Vec<Float>)> {
    let basis: Vec<Vec<Float>> = basis.extract()?;
    let basis: Vec<[Float; 3]> = basis.iter().map(|x| [x[0], x[1], x[2]]).collect();
    let (basis, mapping) = _lll(&basis, delta);
    Ok((basis, mapping))
}

#[pyfunction]
#[pyo3(name = "pbc_all_distances")]
#[pyo3(signature = (lattice, frac_coords))]
fn pbc_all_distances_py(
    lattice: &Bound<'_, PySequence>,
    frac_coords: &Bound<'_, PySequence>,
) -> PyResult<Vec<Vec<Float>>> {
    let lattice: Vec<Vec<Float>> = lattice.extract()?;
    let frac_coords: Vec<Vec<Float>> = frac_coords.extract()?;
    let lattice: Vec<[Float; 3]> = lattice.iter().map(|x| [x[0], x[1], x[2]]).collect();
    let frac_coords: Vec<[Float; 3]> = frac_coords.iter().map(|x| [x[0], x[1], x[2]]).collect();
    let ret = _pbc(&lattice, &frac_coords);
    match ret {
        Ok(d) => {
            let chunk_size = (d.len() as Float).sqrt() as usize;
            let mut ret_ = Vec::new();
            for chunk in d.chunks(chunk_size) {
                ret_.push(chunk.to_vec());
            }
            Ok(ret_)
        }
        Err(e) => Err(PyValueError::new_err(format!("{}", e))),
    }
}
