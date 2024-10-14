// Copyright 2024 TsumiNa.
// SPDX-License-Identifier: Apache-2.0
mod base;
mod empirical;
mod options;

pub use self::base::*;
pub use self::empirical::*;
pub use self::options::*;

use crate::Float;
use crate::{Crystal, CrystalGeneratorError};

pub trait Generator<T> {
    /// Create a [`Generator`] instance from `spacegroup_num` and `options`.
    fn from_spacegroup_num(
        spacegroup_num: usize,
        volume_of_cell: Float,
        options: Option<T>,
    ) -> Result<Self, CrystalGeneratorError>
    where
        Self: Sized;

    /// Return crystal structure.
    fn gen(
        &self,
        elements: &Vec<String>,
        wyckoff_letters: &Vec<String>,
        check_distance: Option<bool>,
        distance_scale_factor: Option<Float>,
    ) -> Result<Crystal, CrystalGeneratorError>;
}
