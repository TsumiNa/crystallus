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

use super::options::BaseGeneratorOption;
use crate::utils::{check_distance as check_distance_, create_lattice, get_multiplied_length};
use crate::Generator;

use crate::structure::{Crystal, CrystalGeneratorError, LatticeFn};

use crate::{wyckoff_pos::*, Float, SPG_TYPES, WY};

use ndarray::{concatenate, Array2, ArrayView2, Axis};
use rand::{thread_rng, Rng};
use rand_distr::{Normal, Uniform};
use std::collections::HashMap;

// #[derive(Debug, Clone, PartialEq)]
pub struct BaseGenerator {
    pub spacegroup_num: usize,
    pub max_attempts_number: u16,
    pub verbose: bool,
    wy_pos_generator: HashMap<String, (usize, WyckoffPos)>,
    lattice_gen: LatticeFn,
}

impl<'a> Generator<BaseGeneratorOption> for BaseGenerator {
    fn from_spacegroup_num(
        spacegroup_num: usize,
        volume_of_cell: Float,
        options: Option<BaseGeneratorOption>,
    ) -> Result<BaseGenerator, CrystalGeneratorError> {
        // check spacegroup number is valid
        if !(1..=230).contains(&spacegroup_num) {
            return Err(CrystalGeneratorError(
                "space group number is illegal".to_owned(),
            ));
        }

        // make options
        let options = match options {
            Some(options) => options,
            None => BaseGeneratorOption::default(),
        };

        // set identical sites
        let verbose = options.verbose;
        let identical_sites = match &SPG_TYPES[spacegroup_num - 1] {
            'A' => vec![[0., 1. / 2., 1. / 2.]],
            'B' => vec![[1. / 2., 0., 1. / 2.]],
            'C' => vec![[1. / 2., 1. / 2., 0.]],
            'I' => vec![[1. / 2., 1. / 2., 1. / 2.]],
            'S' => vec![[1. / 3., 1. / 3., 2. / 3.], [2. / 3., 2. / 3., 1. / 3.]],
            'T' => vec![[1. / 3., 2. / 3., 1. / 3.], [2. / 3., 1. / 3., 2. / 3.]],
            'F' => vec![
                [0., 1. / 2., 1. / 2.],
                [1. / 2., 0., 1. / 2.],
                [1. / 2., 1. / 2., 0.],
            ],
            _ => vec![],
        };

        // set volume and variance
        let variance_of_volume = options.variance_of_volume;
        let multiple = (identical_sites.len() + 1) as Float;
        let volume_of_cell = volume_of_cell * multiple;
        let variance_of_volume = variance_of_volume * multiple;

        // build wyckoff generators
        let mut wy_pos_generator: HashMap<String, (usize, WyckoffPos)> = HashMap::new();
        WY[spacegroup_num - 1].iter().for_each(|(k, (m, _, p))| {
            wy_pos_generator.insert(
                (*k).to_owned(),
                (
                    *m,
                    WyckoffPos::from_str_and_shifts(p, identical_sites.clone()).unwrap(),
                ),
            );
        });

        // set value for conditional parameters
        let angle_tolerance = options.angle_tolerance;
        let angle_range = options.angle_range;

        if angle_range.0 * 3. >= 360. {
            return Err(CrystalGeneratorError(
                "angle range is illegal, sum of min bound must smaller than 360 degree".to_owned(),
            ));
        }

        let (low_length, high_length) = (
            volume_of_cell.powf(1. / 3.) * 0.5,
            volume_of_cell.powf(1. / 3.) * 1.5,
        );
        let length_dist = Uniform::new_inclusive(low_length, high_length);
        let angle_dist = Uniform::new_inclusive(angle_range.0, angle_range.1);
        let volume_dist = Normal::new(volume_of_cell, variance_of_volume).unwrap();

        let max_attempts_number = options.max_attempts_number;

        // note that closure in Rust is not a available type
        // we have to use Box to wrap and save closure
        let lattice_gen: LatticeFn = match spacegroup_num {
            // Triclinic, α≠β≠γ≠；a≠b≠c
            1..=2 => Box::new(move || {
                let vol: Float = thread_rng().sample(volume_dist).abs();

                for _ in 0..max_attempts_number {
                    let angles: Vec<Float> = thread_rng().sample_iter(angle_dist).take(3).collect();
                    // gen angles conditional
                    let sum = angles.iter().sum::<Float>();
                    if angle_tolerance < sum
                        && sum < 360. - angle_tolerance
                        && angles
                            .iter()
                            .filter(|&&x| x > (sum / 2. - angle_tolerance))
                            .count()
                            == 0
                    {
                        let mut abc: Vec<Float> =
                            thread_rng().sample_iter(length_dist).take(2).collect();
                        // c = a*b*c / a*b
                        let c: Float =
                            get_multiplied_length(&vol, &angles) / abc.iter().product::<Float>();

                        if low_length < c && c < high_length {
                            abc.push(c);
                            return Ok((create_lattice(&abc, &angles), vol));
                        }
                    }
                }
                return Err(CrystalGeneratorError(
                    "reached the max attempts (in lattice generation)".to_owned(),
                ));
            }),
            // Monoclinic, α=γ=90°，β≠90°；a≠b≠c
            3..=15 => Box::new(move || {
                let vol: Float = thread_rng().sample(volume_dist).abs();

                // gen angles conditional
                let mut angles = vec![90 as Float, 0., 90.];
                for _ in 0..max_attempts_number {
                    let beta = thread_rng().sample(angle_dist);

                    if beta != 90. {
                        angles[1] = beta;
                        let mut abc: Vec<Float> =
                            thread_rng().sample_iter(length_dist).take(2).collect();
                        // c = a*b*c / a*b
                        let c: Float =
                            get_multiplied_length(&vol, &angles) / abc.iter().product::<Float>();

                        if low_length < c && c < high_length {
                            abc.push(c);
                            return Ok((create_lattice(&abc, &angles), vol));
                        }
                    }
                }
                return Err(CrystalGeneratorError(
                    "reached the max attempts (in lattice generation)".to_owned(),
                ));
            }),
            // Orthorhombic, α=β=γ=90°；a≠b≠c
            16..=74 => Box::new(move || {
                let vol: Float = thread_rng().sample(volume_dist).abs();

                // gen angles conditional
                let angles = vec![90 as Float; 3];
                for _ in 0..max_attempts_number {
                    let mut abc: Vec<Float> =
                        thread_rng().sample_iter(length_dist).take(2).collect();
                    // c = a*b*c / a*b
                    let c: Float = vol / abc.iter().product::<Float>();

                    if low_length < c && c < high_length {
                        abc.push(c);
                        return Ok((create_lattice(&abc, &angles), vol));
                    }
                }
                return Err(CrystalGeneratorError(
                    "reached the max attempts (in lattice generation)".to_owned(),
                ));
            }),
            // Tetragonal, α=β=γ=90°；a=b≠c
            75..=142 => Box::new(move || {
                let vol: Float = thread_rng().sample(volume_dist).abs();

                // gen angles conditional
                let angles = vec![90 as Float; 3];
                for _ in 0..max_attempts_number {
                    let a = thread_rng().sample(length_dist);
                    let c: Float = vol / (a * a);

                    if low_length < c && c < high_length {
                        let abc = vec![a, a, c];
                        return Ok((create_lattice(&abc, &angles), vol));
                    }
                }
                return Err(CrystalGeneratorError(
                    "reached the max attempts (in lattice generation)".to_owned(),
                ));
            }),
            // Trigonal
            n @ 143..=167 => Box::new(move || {
                let vol: Float = thread_rng().sample(volume_dist).abs();
                if ![146, 148, 155, 160, 161, 166, 167].contains(&n) {
                    // α=β=90°，γ=120°；a=b≠c
                    // gen angles conditional
                    let angles = vec![90 as Float, 90., 120.];
                    for _ in 0..max_attempts_number {
                        let a = thread_rng().sample(length_dist);
                        // c = V / a^2 sinγ
                        let sin_gamma = (120 as Float).to_radians().sin();
                        let c: Float = vol / (a * a * sin_gamma);

                        if low_length < c && c < high_length {
                            let abc = vec![a, a, c];
                            return Ok((create_lattice(&abc, &angles), vol));
                        }
                    }
                    return Err(CrystalGeneratorError(
                        "reached the max attempts (in lattice generation)".to_owned(),
                    ));
                } else {
                    // α=β=γ<90°；a=b=c
                    for _ in 0..max_attempts_number {
                        // gen angles conditional
                        let angles =
                            vec![
                                thread_rng()
                                    .sample(Uniform::new(angle_range.0, angle_range.1.min(90.)));
                                3
                            ];
                        let c: Float = get_multiplied_length(&vol, &angles).powf(1. / 3.);

                        if low_length < c && c < high_length {
                            let abc = vec![c, c, c];
                            return Ok((create_lattice(&abc, &angles), vol));
                        }
                    }
                    return Err(CrystalGeneratorError(
                        "reached the max attempts (in lattice generation)".to_owned(),
                    ));
                }
            }),
            // Hexagonal, α=β=90°，γ=120°；a=b≠c
            168..=194 => Box::new(move || {
                let vol: Float = thread_rng().sample(volume_dist).abs();
                // gen angles conditional
                let angles = vec![90 as Float, 90., 120.];
                for _ in 0..max_attempts_number {
                    let a = thread_rng().sample(length_dist);
                    // c = V / a^2 sinγ
                    let sin_gamma = (120 as Float).to_radians().sin();
                    let c: Float = vol / (a * a * sin_gamma);

                    if low_length < c && c < high_length {
                        let abc = vec![a, a, c];
                        return Ok((create_lattice(&abc, &angles), vol));
                    }
                }
                return Err(CrystalGeneratorError(
                    "reached the max attempts (in lattice generation)".to_owned(),
                ));
            }),
            // Cubic, α=β=γ=90°；a=b=c
            195..=230 => Box::new(move || {
                let vol: Float = thread_rng().sample(volume_dist).abs();

                // gen angles conditional
                let angles = vec![90 as Float; 3];
                let c: Float = vol.powf(1. / 3.);
                let abc = vec![c, c, c];

                return Ok((create_lattice(&abc, &angles), vol));
            }),
            // others
            _ => return Err(CrystalGeneratorError("unknown error".to_owned())),
        };

        // return self
        Ok(BaseGenerator {
            spacegroup_num,
            wy_pos_generator,
            verbose,
            lattice_gen,
            max_attempts_number,
        })
    }

    fn gen(
        &self,
        elements: &Vec<String>,
        wyckoff_letters: &Vec<String>,
        check_distance: Option<bool>,
        distance_scale_factor: Option<Float>,
    ) -> Result<Crystal, CrystalGeneratorError> {
        // set default value for options
        let check_distance = check_distance.unwrap_or(true);
        let distance_scale_factor = distance_scale_factor.unwrap_or(0.1);

        // prepare containers
        let mut elements_: Vec<String> = Vec::new();
        let mut wyckoff_letters_: Vec<String> = Vec::new();
        let mut wy_gens_: Vec<&WyckoffPos> = Vec::new(); //  [(wyckoff_pos_gen, wyckoff_letter)...]

        // iter all wyckoff letters and elements
        for (e, w) in elements.iter().zip(wyckoff_letters) {
            match self.wy_pos_generator.get(w) {
                Some((multiplicity, wy_gen)) => {
                    elements_.append(&mut vec![e.to_string(); *multiplicity]);
                    wyckoff_letters_.append(&mut vec![w.to_string(); *multiplicity]);
                    wy_gens_.push(wy_gen);
                }
                None => {
                    return Err(CrystalGeneratorError(format!(
                        "space group {} dose not has wyckoff letter [{}]",
                        self.spacegroup_num, w
                    )))
                }
            }
        }
        // generate lengths and angles, then transfer to a lattice object
        let (lattice, vol) = (self.lattice_gen)()?;

        // generate particles for each element, respectively
        let mut all_particles: Vec<Array2<Float>> = Vec::new();
        for g in wy_gens_.iter() {
            let particles = g.random_gen();
            all_particles.push(particles)
        }

        // join all generated particles in their generated order
        let particles = concatenate(
            Axis(0),
            &all_particles
                .iter()
                .map(|p| p.view())
                .collect::<Vec<ArrayView2<Float>>>()[..],
        )
        .unwrap();

        // check distances between all particles,
        // if ok, return generated crystal object
        if !check_distance {
            return Ok(Crystal {
                spacegroup_num: self.spacegroup_num,
                elements: elements_,
                wyckoff_letters: wyckoff_letters_,
                volume: vol,
                lattice,
                particles,
            });
        } else {
            if check_distance_(&lattice, &particles, &elements_, &distance_scale_factor) {
                return Ok(Crystal {
                    spacegroup_num: self.spacegroup_num,
                    elements: elements_,
                    wyckoff_letters: wyckoff_letters_,
                    volume: vol,
                    lattice,
                    particles,
                });
            }
            return Err(
                CrystalGeneratorError(format!(
                    "Atomic distance check failed for structure: \n\nLattice: {:#.4},\nAtoms: {:#.4}\nVolume: {:.4}\n\nIf you tried many times and still get this error, please try to set `volume_of_cell` and/or `distance_scale_factor` bigger. (in crystal structure generation)",
                    lattice, particles, vol
                ),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_ulps_eq};
    use core::f32::consts::PI;
    use ndarray::{arr2, s};

    fn lattice_to(lattice: Array2<Float>) -> (Vec<Float>, Vec<Float>) {
        fn abs_cap(val: Float) -> Float {
            val.min(1.).max(-1.)
        }

        let abc = lattice
            .mapv(|a| a.powi(2))
            .sum_axis(Axis(1))
            .mapv(f64::sqrt)
            .into_raw_vec();

        let mut angles = vec![0. as Float, 0., 0.];
        for i in 0..3 {
            let j = (i + 1) % 3;
            let k = (i + 2) % 3;
            let mj = lattice.slice(s![j, ..]);
            let mk = lattice.slice(s![k, ..]);
            angles[i] = abs_cap(mj.dot(&mk) / (abc[j] * abc[k]));
        }
        angles = angles
            .into_iter()
            .map(|x| x.acos() * 180.0 / PI as Float)
            .collect();
        (abc, angles)
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: CrystalGeneratorError(\"space group number is illegal\")"
    )]
    fn should_panic_when_using_wrong_spacegroup_number() {
        BaseGenerator::from_spacegroup_num(
            300,
            100.,
            Some(BaseGeneratorOption {
                variance_of_volume: 10.,
                ..Default::default()
            }),
        )
        .unwrap();
    }

    #[test]
    fn should_return_space_group_illegal_err() {
        let tmp = BaseGenerator::from_spacegroup_num(
            300,
            100.,
            Some(BaseGeneratorOption {
                variance_of_volume: 10.,
                ..Default::default()
            }),
        );
        match tmp {
            Err(e) => assert_eq!(
                format!("{}", e),
                "CrystalGeneratorError -- `space group number is illegal`"
            ),
            _ => assert!(false),
        }
    }

    #[test]
    fn should_return_angle_range_illegal_err() {
        let tmp = BaseGenerator::from_spacegroup_num(
            1,
            100.,
            Some(BaseGeneratorOption {
                variance_of_volume: 10.,
                angle_range: (160., 170.),
                ..Default::default()
            }),
        );

        match tmp {
            Err(e) => assert_eq!(
                format!("{}", e),
                "CrystalGeneratorError -- `angle range is illegal, sum of min bound must smaller than 360 degree`"
            ),
            _ => assert!(false),
        }
    }
    #[test]
    #[should_panic(expected = "reached the max attempts (in lattice generation)")]
    fn should_panic_when_reach_max_attempts_number() {
        let tmp = BaseGenerator::from_spacegroup_num(
            2,
            1000.,
            Some(BaseGeneratorOption {
                variance_of_volume: 10.,
                angle_range: (119.999, 200.),
                ..Default::default()
            }),
        )
        .unwrap();
        (tmp.lattice_gen)().unwrap();
    }

    #[test]
    fn create_crystal_generator_from_spacegroup_with_option() -> Result<(), CrystalGeneratorError> {
        let tmp = BaseGenerator::from_spacegroup_num(
            2,
            1000.,
            Some(BaseGeneratorOption {
                variance_of_volume: 0.,
                angle_range: (70., 71.),
                ..Default::default()
            }),
        )
        .unwrap();

        let (lattice, vol) = (tmp.lattice_gen)()?;
        let (abc, angles) = lattice_to(lattice);

        assert_ulps_eq!(vol, 1000.);
        assert!((5. < abc[0]) & (abc[0] < 15.));
        assert!((5. < abc[1]) & (abc[1] < 15.));
        assert!((5. < abc[2]) & (abc[2] < 15.));
        assert!((70. < angles[0]) & (angles[0] < 71.));
        assert!((70. < angles[1]) & (angles[1] < 71.));
        assert!((70. < angles[2]) & (angles[2] < 71.));

        Ok(())
    }

    #[test]
    fn crystal_generator_gen_lattice() -> Result<(), CrystalGeneratorError> {
        let tmp = BaseGenerator::from_spacegroup_num(
            200,
            1000.,
            Some(BaseGeneratorOption {
                variance_of_volume: 10.,

                ..Default::default()
            }),
        )
        .unwrap();
        let (lattice, _vol) = (tmp.lattice_gen)()?;
        let (abc, angles) = lattice_to(lattice);
        assert!(abc.len() == 3);
        assert!(abc.iter().eq(abc.iter()), "{}", true);
        assert!(angles.len() == 3);
        assert!(angles.iter().eq(angles.iter()), "{}", true);
        assert!(9.5 < abc[0] && abc[0] < 10.5);
        assert_abs_diff_eq!(angles[0], 90., epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn crystal_generate() -> Result<(), CrystalGeneratorError> {
        /*
        space group 63
        ======================================================
        Multiplicity | Wyckoff letter |      Coordinates
        ------------------------------------------------------
             4       |         b      | (0,1/2,0) (0,1/2,1/2)
        ------------------------------------------------------
             4       |         a      | (0,0,0) (0,0,1/2)
        ------------------------------------------------------
        */
        let cg = BaseGenerator::from_spacegroup_num(
            63,
            1000.,
            Some(BaseGeneratorOption {
                variance_of_volume: 10.,

                ..Default::default()
            }),
        )?;
        let cry = cg.gen(
            &vec!["Li".to_owned(), "P".to_owned()],
            &vec!["a".to_owned(), "b".to_owned()],
            None,
            None,
        )?;
        assert_eq!(
            cry.elements,
            vec![
                "Li".to_owned(),
                "Li".to_owned(),
                "Li".to_owned(),
                "Li".to_owned(),
                "P".to_owned(),
                "P".to_owned(),
                "P".to_owned(),
                "P".to_owned()
            ]
        );
        assert_eq!(cry.particles.shape(), [8, 3]);
        assert_eq!(
            cry.particles,
            arr2(&[
                [0., 0., 0.],    // Li
                [0., 0., 0.5],   // Li
                [0.5, 0.5, 0.],  // Li
                [0.5, 0.5, 0.5], // Li
                [0., 0.5, 0.],   // P
                [0., 0.5, 0.5],  // P
                [0.5, 1., 0.],   // P
                [0.5, 1., 0.5]   // P
            ])
        );
        Ok(())
    }
}
