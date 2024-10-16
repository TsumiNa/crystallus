// Copyright 2024 TsumiNa
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use itertools::Itertools;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::HashMap;

use libcrystal::{Float, WyckoffCfgGenerator as wyckoff_cfg_gen};

#[pyclass(module = "core")]
pub struct WyckoffCfgGenerator {
    composition: BTreeMap<String, Float>,
    priority: HashMap<usize, HashMap<String, Float>>,

    #[pyo3(get, set)]
    n_jobs: i16,
    #[pyo3(get, set)]
    max_recurrent: u16,
    #[pyo3(get, set)]
    verbose: bool,
}

#[pymethods]
impl WyckoffCfgGenerator {
    #[new]
    #[pyo3( signature = (
        composition,
        *,
        max_recurrent = 1_000,
        n_jobs = -1,
        priority = None,
        verbose = false
    ))]
    fn new(
        composition: &Bound<'_, PyDict>,
        max_recurrent: Option<u16>,
        n_jobs: Option<i16>,
        priority: Option<&Bound<'_, PyDict>>,
        verbose: Option<bool>,
    ) -> PyResult<Self> {
        // convert Option<T: FromPyObject> -> Option<D>
        // if T.extract() return Err(e), pass this panic to python side
        let priority: HashMap<usize, HashMap<String, Float>> = match priority {
            Some(t) => t.extract()?,
            _ => HashMap::new(),
        };

        let composition: BTreeMap<String, Float> = composition.extract()?;
        Ok(WyckoffCfgGenerator {
            max_recurrent: max_recurrent.unwrap_or(1000),
            priority,
            composition,
            n_jobs: n_jobs.unwrap_or(-1),
            verbose: verbose.unwrap_or(false),
        })
    }

    #[pyo3(signature = (spacegroup_num))]
    fn gen_one(&self, py: Python<'_>, spacegroup_num: usize) -> PyResult<PyObject> {
        let priority = match self.priority.get(&spacegroup_num) {
            Some(h) => Some(h.clone()),
            _ => None,
        };
        let wy = wyckoff_cfg_gen::from_spacegroup_num(
            spacegroup_num,
            Some(self.max_recurrent),
            priority,
            self.verbose,
        );
        match wy {
            Ok(wy) => match wy.gen(&self.composition) {
                Err(e) => Err(PyValueError::new_err(e.to_string())),
                Ok(w) => Ok(w.into_py(py)),
            },
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    #[pyo3(signature = (size, /, *spacegroup_num))]
    fn gen_many<'py>(
        &self,
        py: Python<'py>,
        size: i32,
        spacegroup_num: &Bound<'py, PyTuple>,
    ) -> PyResult<PyObject> {
        let spacegroup_num: Vec<usize> = match spacegroup_num.extract() {
            Ok(m) => m,
            Err(_) => {
                return Err(PyValueError::new_err(
                    "`spacegroup_num`s must be an int between 1 - 230",
                ))
            }
        };
        // parallel using rayon
        if self.n_jobs > 0 {
            std::env::set_var("RAYON_NUM_THREADS", self.n_jobs.to_string());
        }
        match spacegroup_num.len() {
            0 => {
                return Err(PyValueError::new_err("no configurations for generation"));
            }
            1 => {
                let sp_num = spacegroup_num[0];
                let priority = match self.priority.get(&sp_num) {
                    Some(h) => Some(h.clone()),
                    _ => None,
                };
                let wy = match wyckoff_cfg_gen::from_spacegroup_num(
                    sp_num,
                    Some(self.max_recurrent),
                    priority,
                    self.verbose,
                ) {
                    Ok(wy) => wy,
                    Err(e) => return Err(PyValueError::new_err(e.to_string())),
                };
                //Do works
                let ret: Vec<BTreeMap<String, Vec<String>>> = py.allow_threads(|| {
                    (0..size)
                        .into_par_iter()
                        .map(|_| wy.gen(&self.composition))
                        .filter_map(Result::ok)
                        .collect()
                });
                let ret: Vec<PyObject> = ret
                    .into_iter()
                    .unique()
                    .map(|cfg| cfg.into_py(py))
                    .collect();
                std::env::set_var("RAYON_NUM_THREADS", "");
                Ok(ret.into_py(py))
            }
            _ => {
                let dict = PyDict::new_bound(py);
                let mut tmp: Vec<wyckoff_cfg_gen> = Vec::new();
                for sp_num in spacegroup_num.iter() {
                    let priority = match self.priority.get(&sp_num) {
                        Some(h) => Some(h.clone()),
                        _ => None,
                    };
                    let wy = match wyckoff_cfg_gen::from_spacegroup_num(
                        *sp_num,
                        Some(self.max_recurrent),
                        priority,
                        self.verbose,
                    ) {
                        Ok(wy) => wy,
                        Err(e) => return Err(PyValueError::new_err(e.to_string())),
                    };
                    tmp.push(wy);
                }
                for (wy, sp_num) in tmp.iter().zip(spacegroup_num) {
                    //Do works
                    let ret: Vec<BTreeMap<String, Vec<String>>> = py.allow_threads(move || {
                        (0..size)
                            .into_par_iter()
                            .map(|_| wy.gen(&self.composition))
                            .filter_map(Result::ok)
                            .collect()
                    });
                    let ret: Vec<PyObject> = ret
                        .into_iter()
                        .unique()
                        .map(|cfg| cfg.into_py(py))
                        .collect();
                    dict.set_item(sp_num, ret)?;
                }
                std::env::set_var("RAYON_NUM_THREADS", "");
                Ok(dict.into_py(py))
            }
        }
    }
}
