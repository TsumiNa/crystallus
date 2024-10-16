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

use either::{Either, Left, Right};
use libcrystal::Generator;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rand::{prelude::SliceRandom, thread_rng};
use rayon::prelude::*;
use std::collections::BTreeMap;

use libcrystal::{
    BaseGenerator, BaseGeneratorOption, Crystal, EmpiricalGenerator, EmpiricalGeneratorOption,
    Float,
};

#[pyclass(module = "core")]
pub struct CrystalGenerator {
    _crystal_gen: Either<BaseGenerator, EmpiricalGenerator>,

    #[pyo3(get, set)]
    n_jobs: i16,
    #[pyo3(get)]
    spacegroup_num: usize,
    #[pyo3(get)]
    max_attempts_number: u16,
}

#[pymethods]
impl CrystalGenerator {
    #[new]
    #[pyo3(
        signature = (
            spacegroup_num,
            volume_of_cell,
            variance_of_volume, *,
            angle_range=(30., 150.),
            angle_tolerance=20.,
            lattice=None,
            empirical_coords = None,
            empirical_coords_variance = 0.01,
            empirical_coords_sampling_rate = 1.,
            empirical_coords_loose_sampling = true,
            max_attempts_number=5_000,
            n_jobs = -1,
            verbose = true
        )
    )]
    fn new(
        spacegroup_num: usize,
        volume_of_cell: Float,
        variance_of_volume: Float,
        angle_range: (Float, Float),
        angle_tolerance: Float,
        lattice: Option<&Bound<'_, PyTuple>>,
        empirical_coords: Option<&Bound<'_, PyTuple>>,
        empirical_coords_variance: Float,
        empirical_coords_sampling_rate: Float,
        empirical_coords_loose_sampling: bool,
        max_attempts_number: u16,
        n_jobs: i16,
        verbose: bool,
    ) -> PyResult<Self> {
        if lattice.is_none() && empirical_coords.is_none() {
            let _crystal_gen = BaseGenerator::from_spacegroup_num(
                spacegroup_num,
                volume_of_cell,
                Some(BaseGeneratorOption {
                    variance_of_volume,
                    angle_range,
                    angle_tolerance,
                    max_attempts_number,
                    verbose,
                }),
            );
            match _crystal_gen {
                Err(e) => Err(PyValueError::new_err(e.to_string())),
                Ok(w) => {
                    return Ok(CrystalGenerator {
                        _crystal_gen: Left(w),
                        n_jobs,
                        spacegroup_num,
                        max_attempts_number,
                    })
                }
            }
        } else {
            // convert Option<T: FromPyObject> -> Option<D>
            // if T.extract() return Err(e), pass this panic to python side
            let empirical_coords: Vec<(String, Vec<Float>)> = match empirical_coords {
                Some(t) => t.extract()?,
                _ => Vec::new(),
            };
            let lattice: Vec<Float> = match lattice {
                Some(t) => {
                    let ret: Vec<Float> = t.extract()?;
                    if ret.len() != 9 {
                        return Err(PyValueError::new_err("`lattice` is illegal"));
                    }
                    ret
                }
                _ => vec![0.; 9],
            };
            let _crystal_gen = EmpiricalGenerator::from_spacegroup_num(
                spacegroup_num,
                volume_of_cell,
                Some(EmpiricalGeneratorOption {
                    base_option: BaseGeneratorOption {
                        variance_of_volume,
                        angle_range,
                        angle_tolerance,
                        max_attempts_number,
                        verbose,
                    },
                    lattice,
                    empirical_coords,
                    empirical_coords_variance,
                    empirical_coords_loose_sampling,
                    empirical_coords_sampling_rate,
                }),
            );
            match _crystal_gen {
                Err(e) => Err(PyValueError::new_err(e.to_string())),
                Ok(w) => {
                    return Ok(CrystalGenerator {
                        _crystal_gen: Right(w),
                        n_jobs,
                        spacegroup_num,
                        max_attempts_number,
                    })
                }
            }
        }
    }

    #[pyo3(signature = (wyckoff_cfg, *, check_distance=true, distance_scale_factor=0.1))]
    fn gen_one<'py>(
        &self,
        py: Python<'py>,
        wyckoff_cfg: &Bound<'py, PyDict>,
        check_distance: bool,
        distance_scale_factor: Float,
    ) -> PyResult<PyObject> {
        let mut cfg: BTreeMap<String, Vec<String>> = match wyckoff_cfg.extract() {
            Ok(m) => m,
            Err(err) => {
                return Err(PyValueError::new_err(format!(
                    "can not converting `cfg` from python, error is <{}>",
                    err
                )))
            }
        };
        // let mut cfg: BTreeMap<String, Vec<String>> = wyckoff_cfg.extract()?;
        let mut elements: Vec<String> = Vec::new();
        let mut wyckoff_letters: Vec<String> = Vec::new();
        for (elem, letter) in cfg.iter_mut() {
            elements.append(&mut vec![(*elem).clone(); letter.len()]);
            wyckoff_letters.append(letter);
        }

        let cry = match &self._crystal_gen {
            Left(l) => l.gen(
                &elements,
                &wyckoff_letters,
                Some(check_distance),
                Some(distance_scale_factor),
            ),
            Right(r) => r.gen(
                &elements,
                &wyckoff_letters,
                Some(check_distance),
                Some(distance_scale_factor),
            ),
        };

        match cry {
            Err(e) => Err(PyValueError::new_err(e.to_string())),
            Ok(w) => {
                let dict = PyDict::new_bound(py);
                dict.set_item("spacegroup_num", w.spacegroup_num)?;
                dict.set_item("volume", w.volume)?;
                dict.set_item(
                    "lattice",
                    w.lattice
                        .into_raw_vec()
                        .chunks(3)
                        .collect::<Vec<&[Float]>>(),
                )?;
                dict.set_item("species", w.elements)?;
                dict.set_item("wyckoff_letters", w.wyckoff_letters)?;
                dict.set_item(
                    "coords",
                    w.particles
                        .into_raw_vec()
                        .chunks(3)
                        .collect::<Vec<&[Float]>>(),
                )?;

                Ok(dict.into_py(py))
            }
        }
    }

    #[pyo3(
        signature = (
            expect_size,
            wyckoff_cfgs,
            *,
            max_attempts=None,
            check_distance=true,
            distance_scale_factor=0.1
        )
    )]
    fn gen_many<'py>(
        &self,
        py: Python<'py>,
        expect_size: usize,
        wyckoff_cfgs: &Bound<'py, PyTuple>,
        max_attempts: Option<usize>,
        check_distance: bool,
        distance_scale_factor: Float,
    ) -> PyResult<PyObject> {
        let mut cfgs: Vec<BTreeMap<String, Vec<String>>> =
            match wyckoff_cfgs.extract() {
                Ok(m) => m,
                Err(err) => return Err(PyValueError::new_err(
                    format!("can not converting `cfg` into dict, make sure the `cfgs` are tuple of dicts, error msg is: <{}>", err),
                )),
            };
        // parallel using rayon
        if self.n_jobs > 0 {
            std::env::set_var("RAYON_NUM_THREADS", self.n_jobs.to_string());
        }

        let mut max_attempts = max_attempts.unwrap_or(expect_size);
        if !check_distance {
            max_attempts = expect_size;
        }
        if max_attempts < expect_size {
            return Err(PyValueError::new_err(
                "`max_attempts` can not be smaller than `expect_size`",
            ));
        }
        let mut ret: Vec<Crystal> = Vec::new();
        match cfgs.len() {
            0 => {
                return Ok(PyTuple::new_bound(py, Vec::<PyDict>::new()).into_py(py));
            }
            1 => {
                let mut elements: Vec<String> = Vec::new();
                let mut wyckoff_letters: Vec<String> = Vec::new();
                for (elem, letter) in cfgs[0].iter_mut() {
                    elements.append(&mut vec![(*elem).clone(); letter.len()]);
                    wyckoff_letters.append(letter);
                }

                //Do works
                ret.append(&mut py.allow_threads(|| {
                    (0..max_attempts)
                        .into_par_iter()
                        .map(|_| match &self._crystal_gen {
                            Left(l) => l.gen(
                                &elements,
                                &wyckoff_letters,
                                Some(check_distance),
                                Some(distance_scale_factor),
                            ),
                            Right(r) => r.gen(
                                &elements,
                                &wyckoff_letters,
                                Some(check_distance),
                                Some(distance_scale_factor),
                            ),
                        })
                        .filter_map(Result::ok)
                        .collect::<Vec<Crystal>>()
                }));

                if ret.len() > expect_size {
                    let mut rng = thread_rng();
                    ret.shuffle(&mut rng);
                    ret = ret.into_iter().take(expect_size).collect();
                }
            }
            _ => {
                for cfg in cfgs.iter_mut() {
                    let mut ret_: Vec<Crystal> = Vec::new();
                    let mut elements: Vec<String> = Vec::new();
                    let mut wyckoff_letters: Vec<String> = Vec::new();
                    for (elem, letter) in cfg.iter_mut() {
                        elements.append(&mut vec![(*elem).clone(); letter.len()]);
                        wyckoff_letters.append(letter);
                    }

                    //Do works
                    ret_.append(&mut py.allow_threads(|| {
                        (0..max_attempts)
                            .into_par_iter()
                            .map(|_| match &self._crystal_gen {
                                Left(l) => l.gen(
                                    &elements,
                                    &wyckoff_letters,
                                    Some(check_distance),
                                    Some(distance_scale_factor),
                                ),
                                Right(r) => r.gen(
                                    &elements,
                                    &wyckoff_letters,
                                    Some(check_distance),
                                    Some(distance_scale_factor),
                                ),
                            })
                            .filter_map(Result::ok)
                            .collect::<Vec<Crystal>>()
                    }));

                    if ret_.len() > expect_size {
                        let mut rng = thread_rng();
                        ret_.shuffle(&mut rng);
                        ret_ = ret_.into_iter().take(expect_size).collect();
                    }
                    ret.append(&mut ret_);
                }
            }
        }

        std::env::set_var("RAYON_NUM_THREADS", "");

        let mut ret_: Vec<PyObject> = Vec::new();
        for crystal in ret {
            let dict = PyDict::new_bound(py);
            dict.set_item("spacegroup_num", crystal.spacegroup_num)?;
            dict.set_item("volume", crystal.volume)?;
            dict.set_item(
                "lattice",
                crystal
                    .lattice
                    .into_raw_vec()
                    .chunks(3)
                    .collect::<Vec<&[Float]>>(),
            )?;
            dict.set_item("species", crystal.elements)?;
            dict.set_item("wyckoff_letters", crystal.wyckoff_letters)?;
            dict.set_item(
                "coords",
                crystal
                    .particles
                    .into_raw_vec()
                    .chunks(3)
                    .collect::<Vec<&[Float]>>(),
            )?;

            ret_.push(dict.into_py(py));
        }
        Ok(PyTuple::new_bound(py, ret_).into_py(py))
    }
}
