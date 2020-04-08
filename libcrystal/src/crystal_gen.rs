// Copyright 2020 TsumiNa. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

mod utils;

use self::utils::pbc_all_distances;
use crate::{wyckoff_pos::*, Float, SPG_TYPES, WY};
use ndarray::{arr2, stack, Array2, ArrayView2, Axis};
use rand::{thread_rng, Rng};
use rand_distr::{Normal, Uniform};
use std::collections::HashMap;
use std::{error, fmt};

// Covalent radius for element (Z=1) H to Cm (Z=96)
const COVALENT_RADIUS: &'static str = std::include_str!("covalent_radius.json");
lazy_static! {
    static ref RADIUS: HashMap<&'static str, Float> =
        serde_json::from_str(COVALENT_RADIUS).unwrap();
}

#[derive(Debug, Clone, PartialEq)]
pub struct Crystal<'a> {
    pub spacegroup_num: usize,
    pub volume: Float,
    pub lattice: Array2<Float>,
    pub particles: Array2<Float>,
    pub elements: Vec<&'a str>,
    pub wyckoff_letters: Vec<&'a str>,
}

#[derive(Debug, Clone)]
pub struct CrystalGeneratorError(String);

impl fmt::Display for CrystalGeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "crystal generator error: `{}`", self.0)
    }
}

impl error::Error for CrystalGeneratorError {}

pub type LatticeFn =
    Box<dyn Fn() -> Result<(Vec<Float>, Vec<Float>, Float), CrystalGeneratorError> + Send + Sync>;
// #[derive(Debug, Clone, PartialEq)]
pub struct CrystalGenerator<'a> {
    pub spacegroup_num: usize,
    pub min_distance_tolerance: Float,
    pub max_recurrent: u16,
    wy_pos_generator: HashMap<&'a str, (usize, WyckoffPos)>,
    lattice_gen: LatticeFn,
}

impl<'a> CrystalGenerator<'a> {
    pub fn from_spacegroup_num(
        spacegroup_num: usize,
        estimated_volume: Float,
        estimated_variance: Float,
        min_distance_tolerance: Option<Float>,
        angle_range: Option<(Float, Float)>,
        angle_tolerance: Option<Float>,
        max_recurrent: Option<u16>,
    ) -> Result<CrystalGenerator<'a>, CrystalGeneratorError> {
        if !(1..=230).contains(&spacegroup_num) {
            return Err(CrystalGeneratorError(
                "space group number is illegal".to_owned(),
            ));
        }
        let shifts = match &SPG_TYPES[spacegroup_num - 1] {
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
        let coefficient = (shifts.len() + 1) as Float;
        let estimated_volume = estimated_volume * coefficient;
        let estimated_variance = estimated_variance * coefficient;

        // build wyckoff generators
        let mut wy_pos_generator: HashMap<&'static str, (usize, WyckoffPos)> = HashMap::new();
        WY[spacegroup_num - 1].iter().for_each(|(k, (m, _, p))| {
            wy_pos_generator.insert(
                *k,
                (
                    *m,
                    WyckoffPos::from_str_and_shifts(p, shifts.clone()).unwrap(),
                ),
            );
        });

        let angle_tolerance = angle_tolerance.unwrap_or(20.);
        let angle_range = angle_range.unwrap_or((30., 150.));
        let min_distance_tolerance = min_distance_tolerance.unwrap_or(0.15);

        if angle_range.0 * 3. >= 360. {
            return Err(CrystalGeneratorError(
                "angle range is illegal, sum of min bound must smaller than 360 degree".to_owned(),
            ));
        }

        let (low_length, high_length) = (
            estimated_volume.powf(1. / 3.) * 0.5,
            estimated_volume.powf(1. / 3.) * 1.5,
        );
        let length_dist = Uniform::new_inclusive(low_length, high_length);
        let angle_dist = Uniform::new_inclusive(angle_range.0, angle_range.1);
        let volume_dist = Normal::new(estimated_volume, estimated_variance).unwrap();

        let max_recurrent = max_recurrent.unwrap_or(5_000);
        let lattice_gen: LatticeFn = match spacegroup_num {
            // Triclinic, α≠β≠γ≠；a≠b≠c
            1..=2 => Box::new(move || {
                let mut rng = thread_rng();
                let vol: Float = rng.sample(volume_dist);

                for _ in 0..max_recurrent {
                    let angles: Vec<Float> = rng.sample_iter(angle_dist).take(3).collect();
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
                        let mut abc: Vec<Float> = rng.sample_iter(length_dist).take(2).collect();
                        // c = a*b*c / a*b
                        let c: Float = CrystalGenerator::get_multiplied_length(&vol, &angles)
                            / abc.iter().product::<Float>();

                        if low_length < c && c < high_length {
                            abc.push(c);
                            return Ok((abc, angles, vol));
                        }
                    }
                }
                return Err(CrystalGeneratorError(
                    "reached the max recurrent number".to_owned(),
                ));
            }),
            // Monoclinic, α=γ=90°，β≠90°；a≠b≠c
            3..=15 => Box::new(move || {
                let mut rng = thread_rng();
                let vol: Float = rng.sample(volume_dist);

                // gen angles conditional
                let mut angles = vec![90 as Float, 0., 90.];
                for _ in 0..max_recurrent {
                    let beta = rng.sample(angle_dist);

                    if beta != 90. {
                        angles[1] = beta;
                        let mut abc: Vec<Float> = rng.sample_iter(length_dist).take(2).collect();
                        // c = a*b*c / a*b
                        let c: Float = CrystalGenerator::get_multiplied_length(&vol, &angles)
                            / abc.iter().product::<Float>();

                        if low_length < c && c < high_length {
                            abc.push(c);
                            return Ok((abc, angles, vol));
                        }
                    }
                }
                return Err(CrystalGeneratorError(
                    "reached the max recurrent number".to_owned(),
                ));
            }),
            // Orthorhombic, α=β=γ=90°；a≠b≠c
            16..=74 => Box::new(move || {
                let mut rng = thread_rng();
                let vol: Float = rng.sample(volume_dist);

                // gen angles conditional
                let angles = vec![90 as Float; 3];
                for _ in 0..max_recurrent {
                    let mut abc: Vec<Float> = rng.sample_iter(length_dist).take(2).collect();
                    // c = a*b*c / a*b
                    let c: Float = vol / abc.iter().product::<Float>();

                    if low_length < c && c < high_length {
                        abc.push(c);
                        return Ok((abc, angles, vol));
                    }
                }
                return Err(CrystalGeneratorError(
                    "reached the max recurrent number".to_owned(),
                ));
            }),
            // Tetragonal, α=β=γ=90°；a=b≠c
            75..=142 => Box::new(move || {
                let mut rng = thread_rng();
                let vol: Float = rng.sample(volume_dist);

                // gen angles conditional
                let angles = vec![90 as Float; 3];
                for _ in 0..max_recurrent {
                    let a = rng.sample(length_dist);
                    let c: Float = vol / a * a;

                    if low_length < c && c < high_length {
                        let abc = vec![a, a, c];
                        return Ok((abc, angles, vol));
                    }
                }
                return Err(CrystalGeneratorError(
                    "reached the max recurrent number".to_owned(),
                ));
            }),
            // Trigonal
            n @ 143..=167 => Box::new(move || {
                let mut rng = thread_rng();
                let vol: Float = rng.sample(volume_dist);
                if ![146, 148, 155, 160, 161, 166, 167].contains(&n) {
                    // α=β=90°，γ=120°；a=b≠c
                    // gen angles conditional
                    let angles = vec![90 as Float, 90., 120.];
                    for _ in 0..max_recurrent {
                        let a = rng.sample(length_dist);
                        // c = V / a^2 sinγ
                        let sin_gamma = (120 as Float).to_radians().sin();
                        let c: Float = vol / (a * a * sin_gamma);

                        if low_length < c && c < high_length {
                            let abc = vec![a, a, c];
                            return Ok((abc, angles, vol));
                        }
                    }
                    return Err(CrystalGeneratorError(
                        "reached the max recurrent number".to_owned(),
                    ));
                } else {
                    // α=β=γ<90°；a=b=c
                    for _ in 0..max_recurrent {
                        // gen angles conditional
                        let angles = vec![rng.sample(Uniform::new(angle_range.0, 90.)); 3];
                        let c: Float =
                            CrystalGenerator::get_multiplied_length(&vol, &angles).powf(1. / 3.);

                        if low_length < c && c < high_length {
                            let abc = vec![c, c, c];
                            return Ok((abc, angles, vol));
                        }
                    }
                    return Err(CrystalGeneratorError(
                        "reached the max recurrent number".to_owned(),
                    ));
                }
            }),
            // Hexagonal, α=β=90°，γ=120°；a=b≠c
            168..=194 => Box::new(move || {
                let mut rng = thread_rng();
                let vol: Float = rng.sample(volume_dist);

                // gen angles conditional
                let angles = vec![90 as Float, 90., 120.];
                for _ in 0..max_recurrent {
                    let a = rng.sample(length_dist);
                    // c = V / a^2 sinγ
                    let sin_gamma = (120 as Float).to_radians().sin();
                    let c: Float = vol / (a * a * sin_gamma);

                    if low_length < c && c < high_length {
                        let abc = vec![a, a, c];
                        return Ok((abc, angles, vol));
                    }
                }
                return Err(CrystalGeneratorError(
                    "reached the max recurrent number".to_owned(),
                ));
            }),
            // Cubic, α=β=γ=90°；a=b=c
            195..=230 => Box::new(move || {
                let mut rng = thread_rng();
                let vol: Float = rng.sample(volume_dist);

                // gen angles conditional
                let angles = vec![90 as Float; 3];
                let c: Float = vol.powf(1. / 3.);
                let abc = vec![c, c, c];

                return Ok((abc, angles, vol));
            }),
            // others
            _ => return Err(CrystalGeneratorError("unknown error".to_owned())),
        };
        Ok(CrystalGenerator {
            min_distance_tolerance,
            spacegroup_num,
            wy_pos_generator,
            lattice_gen,
            max_recurrent,
        })
    }

    #[inline]
    fn get_covalent_radius(element: &Vec<&str>) -> Vec<Float> {
        let mut ret: Vec<Float> = Vec::new();
        for e in element.iter() {
            ret.push(RADIUS[e.clone()]);
        }
        ret
    }
    #[inline]
    fn lattice_from(abc: Vec<Float>, angles: Vec<Float>) -> Array2<Float> {
        let (a, b, c, alpha, beta, gamma) = (
            abc[0],
            abc[1],
            abc[2],
            angles[0].to_radians(),
            angles[1].to_radians(),
            angles[2].to_radians(),
        );
        let (cos_alpha, cos_beta, cos_gamma) = (alpha.cos(), beta.cos(), gamma.cos());

        let (sin_alpha, sin_beta) = (alpha.sin(), beta.sin());

        let gamma_star = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta);

        // Sometimes rounding errors result in values slightly > 1.
        let gamma_star = gamma_star.min(1.).max(-1.).acos();
        let vector_a = [a * sin_beta, 0.0, a * cos_beta];
        let vector_b = [
            -b * sin_alpha * gamma_star.cos(),
            b * sin_alpha * gamma_star.sin(),
            b * cos_alpha,
        ];
        let vector_c = [0.0, 0.0, c];

        arr2(&[vector_a, vector_b, vector_c])
    }

    #[inline]
    pub fn gen(
        &self,
        elements: &Vec<&'a str>,
        wyckoff_letters: &Vec<&'a str>,
    ) -> Result<Crystal, CrystalGeneratorError> {
        let mut elements_: Vec<&'a str> = Vec::new();
        let mut wyckoff_letters_: Vec<&'a str> = Vec::new();
        let mut wy_gens_: Vec<&WyckoffPos> = Vec::new();
        // test all wyckoff letters and build elements list
        for (e, w) in elements.iter().zip(wyckoff_letters) {
            match self.wy_pos_generator.get(w) {
                Some((multiplicity, wy_gen)) => {
                    elements_.append(&mut vec![*e; *multiplicity]);
                    wyckoff_letters_.append(&mut vec![*w; *multiplicity]);
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
        for _ in 0..self.max_recurrent {
            // generate lengths and angles, then transfer to a lattice object
            let (abc, angles, vol) = (self.lattice_gen)()?;
            let lattice = CrystalGenerator::lattice_from(abc, angles);

            // generate particles for each element, respectively
            let mut all_particles: Vec<Array2<Float>> = Vec::new();
            for g in wy_gens_.iter() {
                let particles = g.random_gen();
                all_particles.push(particles)
            }

            // join all generated particles in their generated order
            let particles = stack(
                Axis(0),
                &all_particles
                    .iter()
                    .map(|p| p.view())
                    .collect::<Vec<ArrayView2<Float>>>()[..],
            )
            .unwrap();

            // check distances between all particles,
            // if ok, return generated crystal object
            if self.check_distance(&lattice, &particles, &elements_) {
                return Ok(Crystal {
                    spacegroup_num: self.spacegroup_num,
                    elements: elements_,
                    wyckoff_letters: wyckoff_letters_,
                    volume: vol,
                    lattice,
                    particles,
                });
            }
        }
        return Err(CrystalGeneratorError(
            "reached the max recurrent number".to_owned(),
        ));
    }

    #[inline]
    fn _volume(abc: &Vec<Float>, angles: &Vec<Float>) -> Float {
        let (a, b, c) = (abc[0], abc[1], abc[2]);
        let (cos_alpha, cos_beta, cos_gamma) = (
            angles[0].to_radians().cos(),
            angles[1].to_radians().cos(),
            angles[2].to_radians().cos(),
        );
        a * b
            * c
            * (1. + 2. * cos_alpha * cos_beta * cos_gamma
                - cos_alpha.powi(2)
                - cos_beta.powi(2)
                - cos_gamma.powi(2))
            .sqrt()
    }

    #[inline]
    fn get_multiplied_length(volume: &Float, angles: &Vec<Float>) -> Float {
        let (cos_alpha, cos_beta, cos_gamma) = (
            angles[0].to_radians().cos(),
            angles[1].to_radians().cos(),
            angles[2].to_radians().cos(),
        );

        volume
            / (1. + 2. * cos_alpha * cos_beta * cos_gamma
                - cos_alpha.powi(2)
                - cos_beta.powi(2)
                - cos_gamma.powi(2))
            .sqrt()
    }

    fn check_distance(
        &self,
        lattice: &Array2<Float>,
        particles: &Array2<Float>,
        elements: &Vec<&str>,
    ) -> bool {
        match pbc_all_distances(lattice, particles) {
            Ok(distance_matrix) => {
                let ii = distance_matrix.shape()[0];
                let radius = CrystalGenerator::get_covalent_radius(elements);
                for i in 0..(ii - 1) {
                    for j in (i + 1)..ii {
                        if distance_matrix[[i, j]]
                            < radius[i] + radius[j] - self.min_distance_tolerance
                        {
                            return false;
                        }
                    }
                }

                return true;
            }
            _ => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use ndarray::arr2;
    #[test]
    fn test_get_covalent_radius() {
        let elements = vec!["Li", "P", "O", "Ti", "Pd"];
        assert_eq!(
            CrystalGenerator::get_covalent_radius(&elements),
            vec![1.21, 1.04, 0.64, 1.52, 1.33]
        )
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: CrystalGeneratorError(\"space group number is illegal\")"
    )]
    fn should_panic_when_using_wrong_spacegroup_number() {
        CrystalGenerator::from_spacegroup_num(300, 100., 10., None, None, None, None).unwrap();
    }

    #[test]
    fn should_return_space_group_illegal_err() {
        let tmp = CrystalGenerator::from_spacegroup_num(300, 100., 10., None, None, None, None);
        match tmp {
            Err(e) => assert_eq!(
                format!("{}", e),
                "crystal generator error: `space group number is illegal`"
            ),
            _ => assert!(false),
        }
    }

    #[test]
    fn should_return_angle_range_illegal_err() {
        let tmp = CrystalGenerator::from_spacegroup_num(
            1,
            100.,
            10.,
            None,
            Some((160., 170.)),
            None,
            None,
        );
        match tmp {
            Err(e) => assert_eq!(
                format!("{}", e),
                "crystal generator error: `angle range is illegal, sum of min bound must smaller than 360 degree`"
            ),
            _ => assert!(false),
        }
    }
    #[test]
    #[should_panic(expected = "reached the max recurrent number")]
    fn should_panic_when_reach_max_recurrent() {
        let tmp = CrystalGenerator::from_spacegroup_num(
            2,
            1000.,
            10.,
            Some(5.),
            Some((119.999, 200.)),
            None,
            None,
        )
        .unwrap();
        (tmp.lattice_gen)().unwrap();
    }

    #[test]
    fn test_create_crystal_generator_from_spacegroup() {
        let tmp =
            CrystalGenerator::from_spacegroup_num(2, 100., 10., None, None, None, None).unwrap();
        assert_eq!(tmp.min_distance_tolerance, 0.15);
    }
    #[test]
    fn test_create_crystal_generator_from_spacegroup_with_option(
    ) -> Result<(), CrystalGeneratorError> {
        let tmp = CrystalGenerator::from_spacegroup_num(
            2,
            1000.,
            0.,
            Some(5.),
            Some((70., 71.)),
            None,
            None,
        )
        .unwrap();
        let (abc, angles, vol) = (tmp.lattice_gen)()?;
        assert_ulps_eq!(vol, 1000.);
        assert_eq!(tmp.min_distance_tolerance, 5.);
        assert_eq!(tmp.min_distance_tolerance, 5.);
        assert!((5. < abc[0]) & (abc[0] < 15.));
        assert!((5. < abc[1]) & (abc[1] < 15.));
        assert!((5. < abc[2]) & (abc[2] < 15.));
        assert!((70. < angles[0]) & (angles[0] < 71.));
        assert!((70. < angles[1]) & (angles[1] < 71.));
        assert!((70. < angles[2]) & (angles[2] < 71.));

        Ok(())
    }

    #[test]
    fn test_crystal_generator_gen_lattice() -> Result<(), CrystalGeneratorError> {
        let tmp =
            CrystalGenerator::from_spacegroup_num(200, 1000., 10., None, None, None, None).unwrap();
        let (abc, angles, _vol) = (tmp.lattice_gen)()?;
        assert!(abc.len() == 3);
        assert!(abc.iter().eq(abc.iter()), true);
        assert!(angles.len() == 3);
        assert!(angles.iter().eq(angles.iter()), true);
        assert!(9.5 < abc[0] && abc[0] < 10.5);
        assert!(angles[0] == 90.);
        Ok(())
    }

    #[test]
    fn test_check_distance() {
        let tmp = CrystalGenerator::from_spacegroup_num(
            12,
            145.75949096679688,
            20.,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let lattice = arr2(&[
            [14.019043922424316, 0.0, -6.127918936726928e-07],
            [
                -4.818087404601101e-07,
                6.381750106811523,
                -2.7895515586351394e-07,
            ],
            [0.0, 0.0, 9.891742706298828],
        ]);

        let particles = arr2(&[
            [0., 0.69679058, 0.25],
            [0., -0.69679058, 0.75],
            [0.5, 1.19679058, 0.25],
            [0.5, -0.19679058, 0.75],
            [0., 0.13124257, 0.43134677],
            [0., -0.13124257, 0.93134677],
            [0., 0.13124257, 0.06865323],
            [0., -0.13124257, -0.43134677],
            [0.5, 0.63124257, 0.43134677],
            [0.5, 0.36875743, 0.93134677],
            [0.5, 0.63124257, 0.06865323],
            [0.5, 0.36875743, -0.43134677],
            [0., 0.26911515, 0.7328701],
            [0., -0.26911515, 1.2328701],
            [0., 0.26911515, -0.2328701],
            [0., -0.26911515, -0.7328701],
            [0.5, 0.76911515, 0.7328701],
            [0.5, 0.23088485, 1.2328701],
            [0.5, 0.76911515, -0.2328701],
            [0.5, 0.23088485, -0.7328701],
            [0., 0.03196704, 0.25],
            [0., -0.03196704, 0.75],
            [0.5, 0.53196704, 0.25],
            [0.5, 0.46803296, 0.75],
        ]);
        let elements = vec![
            "Ti", "Ti", "Ti", "Ti", "Ti", "Ti", "Ti", "Ti", "O", "O", "O", "O", "O", "O", "O", "O",
            "O", "O", "O", "O", "O", "O", "O", "O",
        ];
        assert_eq!(tmp.check_distance(&lattice, &particles, &elements), false);
    }

    #[test]
    fn test_crystal_generate() -> Result<(), CrystalGeneratorError> {
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
        let cg = CrystalGenerator::from_spacegroup_num(63, 1000., 10., None, None, None, None)?;
        let cry = cg.gen(&vec!["Li", "P"], &vec!["a", "b"])?;
        assert_eq!(
            cry.elements,
            vec!["Li", "Li", "Li", "Li", "P", "P", "P", "P"]
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
