// Copyright 2024 TsumiNa.
// SPDX-License-Identifier: Apache-2.0

use crate::Float;
use ndarray::Array2;
use std::{error, fmt};

#[derive(Debug, Clone)]
pub struct CrystalGeneratorError(pub String);

impl fmt::Display for CrystalGeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "CrystalGeneratorError -- `{}`", self.0)
    }
}

impl error::Error for CrystalGeneratorError {}

pub type LatticeFn =
    Box<dyn Fn() -> Result<(Array2<Float>, Float), CrystalGeneratorError> + Send + Sync>;

#[derive(Debug, Clone, PartialEq)]
pub struct Crystal {
    pub spacegroup_num: usize,
    pub volume: Float,
    pub lattice: Array2<Float>,
    pub particles: Array2<Float>,
    pub elements: Vec<String>,
    pub wyckoff_letters: Vec<String>,
}
