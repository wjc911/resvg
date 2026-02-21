// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{ImageRef, ImageRefMut, f32_bound};
use rgb::RGBA8;
use usvg::filter::{DiffuseLighting, LightSource, SpecularLighting};
use usvg::{ApproxEqUlps, ApproxZeroUlps, Color};

const FACTOR_1_2: f32 = 1.0 / 2.0;
const FACTOR_1_3: f32 = 1.0 / 3.0;
const FACTOR_1_4: f32 = 1.0 / 4.0;
const FACTOR_2_3: f32 = 2.0 / 3.0;

#[derive(Clone, Copy, Debug)]
struct Vector2 {
    x: f32,
    y: f32,
}

impl Vector2 {
    #[inline]
    fn new(x: f32, y: f32) -> Self {
        Vector2 { x, y }
    }

    #[inline]
    fn approx_zero(&self) -> bool {
        self.x.approx_zero_ulps(4) && self.y.approx_zero_ulps(4)
    }
}

impl core::ops::Mul<f32> for Vector2 {
    type Output = Self;

    #[inline]
    fn mul(self, c: f32) -> Self::Output {
        Vector2 {
            x: self.x * c,
            y: self.y * c,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vector3 {
    #[inline]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Vector3 { x, y, z }
    }

    #[inline]
    fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[inline]
    fn normalized(&self) -> Option<Self> {
        let length = self.length();
        if !length.approx_zero_ulps(4) {
            Some(Vector3 {
                x: self.x / length,
                y: self.y / length,
                z: self.z / length,
            })
        } else {
            None
        }
    }
}

impl core::ops::Add<Vector3> for Vector3 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Vector3) -> Self::Output {
        Vector3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl core::ops::Sub<Vector3> for Vector3 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Vector3) -> Self::Output {
        Vector3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Normal {
    factor: Vector2,
    normal: Vector2,
}

impl Normal {
    #[inline]
    fn new(factor_x: f32, factor_y: f32, nx: i16, ny: i16) -> Self {
        Normal {
            factor: Vector2::new(factor_x, factor_y),
            normal: Vector2::new(-nx as f32, -ny as f32),
        }
    }
}

/// Renders a diffuse lighting.
///
/// - `src` pixels can have any alpha method, since only the alpha channel is used.
/// - `dest` will have an **unpremultiplied alpha**.
///
/// Does nothing when `src` is less than 3x3.
///
/// # Panics
///
/// - When `src` and `dest` have different sizes.
pub fn diffuse_lighting(
    fe: &DiffuseLighting,
    light_source: LightSource,
    src: ImageRef,
    dest: ImageRefMut,
) {
    assert!(src.width == dest.width && src.height == dest.height);

    let light_factor = |normal: Normal, light_vector: Vector3| {
        let k = if normal.normal.approx_zero() {
            light_vector.z
        } else {
            let mut n = normal.normal * (fe.surface_scale() / 255.0);
            n.x *= normal.factor.x;
            n.y *= normal.factor.y;

            let normal = Vector3::new(n.x, n.y, 1.0);

            normal.dot(&light_vector) / normal.length()
        };

        fe.diffuse_constant() * k
    };

    apply(
        light_source,
        fe.surface_scale(),
        fe.lighting_color(),
        &light_factor,
        calc_diffuse_alpha,
        src,
        dest,
    );
}

/// Renders a specular lighting.
///
/// - `src` pixels can have any alpha method, since only the alpha channel is used.
/// - `dest` will have a **premultiplied alpha**.
///
/// Does nothing when `src` is less than 3x3.
///
/// # Panics
///
/// - When `src` and `dest` have different sizes.
pub fn specular_lighting(
    fe: &SpecularLighting,
    light_source: LightSource,
    src: ImageRef,
    dest: ImageRefMut,
) {
    assert!(src.width == dest.width && src.height == dest.height);

    specular_lighting_optimized(fe, light_source, src, dest);
}

/// The original (naive) specular lighting implementation, preserved verbatim.
#[cfg(test)]
fn specular_lighting_naive(
    fe: &SpecularLighting,
    light_source: LightSource,
    src: ImageRef,
    dest: ImageRefMut,
) {
    let light_factor = |normal: Normal, light_vector: Vector3| {
        let h = light_vector + Vector3::new(0.0, 0.0, 1.0);
        let h_length = h.length();

        if h_length.approx_zero_ulps(4) {
            return 0.0;
        }

        let k = if normal.normal.approx_zero() {
            let n_dot_h = h.z / h_length;
            if fe.specular_exponent().approx_eq_ulps(&1.0, 4) {
                n_dot_h
            } else {
                n_dot_h.powf(fe.specular_exponent())
            }
        } else {
            let mut n = normal.normal * (fe.surface_scale() / 255.0);
            n.x *= normal.factor.x;
            n.y *= normal.factor.y;

            let normal = Vector3::new(n.x, n.y, 1.0);

            let n_dot_h = normal.dot(&h) / normal.length() / h_length;
            if fe.specular_exponent().approx_eq_ulps(&1.0, 4) {
                n_dot_h
            } else {
                n_dot_h.powf(fe.specular_exponent())
            }
        };

        fe.specular_constant() * k
    };

    apply(
        light_source,
        fe.surface_scale(),
        fe.lighting_color(),
        &light_factor,
        calc_specular_alpha,
        src,
        dest,
    );
}

/// Optimized specular lighting implementation.
///
/// Key optimizations over the naive version:
/// 1. **Devirtualization / inlining**: the specular factor computation is a
///    monomorphized `#[inline] fn` instead of a `dyn Fn` trait-object closure
///    passed through `apply()`. This lets LLVM inline and optimise the hot path
///    without indirect-call overhead.
/// 2. Specular-exponent == 1.0 boolean (`exp_is_one`) is hoisted out of the
///    per-pixel loop so the branch predictor sees a loop-invariant condition.
/// 3. `surface_scale / 255.0` is pre-computed once (`scale_factor`) instead of
///    being recalculated for every pixel.
///
/// Note: this is *not* a SIMD or data-layout optimisation. The pixel data is
/// still in AoS `RGBA8` order and the per-pixel branching prevents
/// auto-vectorisation.
fn specular_lighting_optimized(
    fe: &SpecularLighting,
    light_source: LightSource,
    src: ImageRef,
    mut dest: ImageRefMut,
) {
    if src.width < 3 || src.height < 3 {
        return;
    }

    let width = src.width;
    let height = src.height;
    let surface_scale = fe.surface_scale();
    let specular_exponent = fe.specular_exponent();
    let specular_constant = fe.specular_constant();
    let lighting_color = fe.lighting_color();
    let exp_is_one = specular_exponent.approx_eq_ulps(&1.0, 4);
    let scale_factor = surface_scale / 255.0;

    // `feDistantLight` has a fixed vector, so calculate it beforehand.
    let mut light_vector = match light_source {
        LightSource::DistantLight(light) => {
            let azimuth = light.azimuth.to_radians();
            let elevation = light.elevation.to_radians();
            Vector3::new(
                azimuth.cos() * elevation.cos(),
                azimuth.sin() * elevation.cos(),
                elevation.sin(),
            )
        }
        _ => Vector3::new(1.0, 1.0, 1.0),
    };

    #[inline]
    fn specular_factor(
        normal: Normal,
        light_vector: Vector3,
        specular_exponent: f32,
        specular_constant: f32,
        exp_is_one: bool,
        scale_factor: f32,
    ) -> f32 {
        let h = light_vector + Vector3::new(0.0, 0.0, 1.0);
        let h_length = h.length();

        if h_length.approx_zero_ulps(4) {
            return 0.0;
        }

        let k = if normal.normal.approx_zero() {
            let n_dot_h = h.z / h_length;
            if exp_is_one {
                n_dot_h
            } else {
                n_dot_h.powf(specular_exponent)
            }
        } else {
            let mut n = normal.normal * scale_factor;
            n.x *= normal.factor.x;
            n.y *= normal.factor.y;

            let normal = Vector3::new(n.x, n.y, 1.0);

            let n_dot_h = normal.dot(&h) / normal.length() / h_length;
            if exp_is_one {
                n_dot_h
            } else {
                n_dot_h.powf(specular_exponent)
            }
        };

        specular_constant * k
    }

    let mut calc = |nx, ny, normal: Normal| {
        match light_source {
            LightSource::DistantLight(_) => {}
            // Note: PointLight and SpotLight arms are identical but cannot be
            // merged with an or-pattern because `PointLight` and `SpotLight`
            // are distinct types — Rust requires a single binding type per arm.
            LightSource::PointLight(ref light) => {
                let nz = src.alpha_at(nx, ny) as f32 / 255.0 * surface_scale;
                let origin = Vector3::new(light.x, light.y, light.z);
                let v = origin - Vector3::new(nx as f32, ny as f32, nz);
                light_vector = v.normalized().unwrap_or(v);
            }
            LightSource::SpotLight(ref light) => {
                let nz = src.alpha_at(nx, ny) as f32 / 255.0 * surface_scale;
                let origin = Vector3::new(light.x, light.y, light.z);
                let v = origin - Vector3::new(nx as f32, ny as f32, nz);
                light_vector = v.normalized().unwrap_or(v);
            }
        }

        let light_color = light_color(&light_source, lighting_color, light_vector);
        let factor = specular_factor(
            normal,
            light_vector,
            specular_exponent,
            specular_constant,
            exp_is_one,
            scale_factor,
        );

        let compute = |x| (f32_bound(0.0, x as f32 * factor, 255.0) + 0.5) as u8;

        let r = compute(light_color.red);
        let g = compute(light_color.green);
        let b = compute(light_color.blue);
        let a = calc_specular_alpha(r, g, b);

        *dest.pixel_at_mut(nx, ny) = RGBA8 { b, g, r, a };
    };

    calc(0, 0, top_left_normal(src));
    calc(width - 1, 0, top_right_normal(src));
    calc(0, height - 1, bottom_left_normal(src));
    calc(width - 1, height - 1, bottom_right_normal(src));

    for x in 1..width - 1 {
        calc(x, 0, top_row_normal(src, x));
        calc(x, height - 1, bottom_row_normal(src, x));
    }

    for y in 1..height - 1 {
        calc(0, y, left_column_normal(src, y));
        calc(width - 1, y, right_column_normal(src, y));
    }

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            calc(x, y, interior_normal(src, x, y));
        }
    }
}

fn apply(
    light_source: LightSource,
    surface_scale: f32,
    lighting_color: Color,
    light_factor: &dyn Fn(Normal, Vector3) -> f32,
    calc_alpha: fn(u8, u8, u8) -> u8,
    src: ImageRef,
    mut dest: ImageRefMut,
) {
    if src.width < 3 || src.height < 3 {
        return;
    }

    let width = src.width;
    let height = src.height;

    // `feDistantLight` has a fixed vector, so calculate it beforehand.
    let mut light_vector = match light_source {
        LightSource::DistantLight(light) => {
            let azimuth = light.azimuth.to_radians();
            let elevation = light.elevation.to_radians();
            Vector3::new(
                azimuth.cos() * elevation.cos(),
                azimuth.sin() * elevation.cos(),
                elevation.sin(),
            )
        }
        _ => Vector3::new(1.0, 1.0, 1.0),
    };

    let mut calc = |nx, ny, normal: Normal| {
        match light_source {
            LightSource::DistantLight(_) => {}
            LightSource::PointLight(ref light) => {
                let nz = src.alpha_at(nx, ny) as f32 / 255.0 * surface_scale;
                let origin = Vector3::new(light.x, light.y, light.z);
                let v = origin - Vector3::new(nx as f32, ny as f32, nz);
                light_vector = v.normalized().unwrap_or(v);
            }
            LightSource::SpotLight(ref light) => {
                let nz = src.alpha_at(nx, ny) as f32 / 255.0 * surface_scale;
                let origin = Vector3::new(light.x, light.y, light.z);
                let v = origin - Vector3::new(nx as f32, ny as f32, nz);
                light_vector = v.normalized().unwrap_or(v);
            }
        }

        let light_color = light_color(&light_source, lighting_color, light_vector);
        let factor = light_factor(normal, light_vector);

        let compute = |x| (f32_bound(0.0, x as f32 * factor, 255.0) + 0.5) as u8;

        let r = compute(light_color.red);
        let g = compute(light_color.green);
        let b = compute(light_color.blue);
        let a = calc_alpha(r, g, b);

        *dest.pixel_at_mut(nx, ny) = RGBA8 { b, g, r, a };
    };

    calc(0, 0, top_left_normal(src));
    calc(width - 1, 0, top_right_normal(src));
    calc(0, height - 1, bottom_left_normal(src));
    calc(width - 1, height - 1, bottom_right_normal(src));

    for x in 1..width - 1 {
        calc(x, 0, top_row_normal(src, x));
        calc(x, height - 1, bottom_row_normal(src, x));
    }

    for y in 1..height - 1 {
        calc(0, y, left_column_normal(src, y));
        calc(width - 1, y, right_column_normal(src, y));
    }

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            calc(x, y, interior_normal(src, x, y));
        }
    }
}

fn light_color(light: &LightSource, lighting_color: Color, light_vector: Vector3) -> Color {
    match *light {
        LightSource::DistantLight(_) | LightSource::PointLight(_) => lighting_color,
        LightSource::SpotLight(ref light) => {
            let origin = Vector3::new(light.x, light.y, light.z);
            let direction = Vector3::new(light.points_at_x, light.points_at_y, light.points_at_z);
            let direction = direction - origin;
            let direction = direction.normalized().unwrap_or(direction);
            let minus_l_dot_s = -light_vector.dot(&direction);
            if minus_l_dot_s <= 0.0 {
                return Color::black();
            }

            if let Some(limiting_cone_angle) = light.limiting_cone_angle {
                if minus_l_dot_s < limiting_cone_angle.to_radians().cos() {
                    return Color::black();
                }
            }

            let factor = minus_l_dot_s.powf(light.specular_exponent.get());
            let compute = |x| (f32_bound(0.0, x as f32 * factor, 255.0) + 0.5) as u8;

            Color::new_rgb(
                compute(lighting_color.red),
                compute(lighting_color.green),
                compute(lighting_color.blue),
            )
        }
    }
}

fn top_left_normal(img: ImageRef) -> Normal {
    let center = img.alpha_at(0, 0);
    let right = img.alpha_at(1, 0);
    let bottom = img.alpha_at(0, 1);
    let bottom_right = img.alpha_at(1, 1);

    Normal::new(
        FACTOR_2_3,
        FACTOR_2_3,
        -2 * center + 2 * right - bottom + bottom_right,
        -2 * center - right + 2 * bottom + bottom_right,
    )
}

fn top_right_normal(img: ImageRef) -> Normal {
    let left = img.alpha_at(img.width - 2, 0);
    let center = img.alpha_at(img.width - 1, 0);
    let bottom_left = img.alpha_at(img.width - 2, 1);
    let bottom = img.alpha_at(img.width - 1, 1);

    Normal::new(
        FACTOR_2_3,
        FACTOR_2_3,
        -2 * left + 2 * center - bottom_left + bottom,
        -left - 2 * center + bottom_left + 2 * bottom,
    )
}

fn bottom_left_normal(img: ImageRef) -> Normal {
    let top = img.alpha_at(0, img.height - 2);
    let top_right = img.alpha_at(1, img.height - 2);
    let center = img.alpha_at(0, img.height - 1);
    let right = img.alpha_at(1, img.height - 1);

    Normal::new(
        FACTOR_2_3,
        FACTOR_2_3,
        -top + top_right - 2 * center + 2 * right,
        -2 * top - top_right + 2 * center + right,
    )
}

fn bottom_right_normal(img: ImageRef) -> Normal {
    let top_left = img.alpha_at(img.width - 2, img.height - 2);
    let top = img.alpha_at(img.width - 1, img.height - 2);
    let left = img.alpha_at(img.width - 2, img.height - 1);
    let center = img.alpha_at(img.width - 1, img.height - 1);

    Normal::new(
        FACTOR_2_3,
        FACTOR_2_3,
        -top_left + top - 2 * left + 2 * center,
        -top_left - 2 * top + left + 2 * center,
    )
}

fn top_row_normal(img: ImageRef, x: u32) -> Normal {
    let left = img.alpha_at(x - 1, 0);
    let center = img.alpha_at(x, 0);
    let right = img.alpha_at(x + 1, 0);
    let bottom_left = img.alpha_at(x - 1, 1);
    let bottom = img.alpha_at(x, 1);
    let bottom_right = img.alpha_at(x + 1, 1);

    Normal::new(
        FACTOR_1_3,
        FACTOR_1_2,
        -2 * left + 2 * right - bottom_left + bottom_right,
        -left - 2 * center - right + bottom_left + 2 * bottom + bottom_right,
    )
}

fn bottom_row_normal(img: ImageRef, x: u32) -> Normal {
    let top_left = img.alpha_at(x - 1, img.height - 2);
    let top = img.alpha_at(x, img.height - 2);
    let top_right = img.alpha_at(x + 1, img.height - 2);
    let left = img.alpha_at(x - 1, img.height - 1);
    let center = img.alpha_at(x, img.height - 1);
    let right = img.alpha_at(x + 1, img.height - 1);

    Normal::new(
        FACTOR_1_3,
        FACTOR_1_2,
        -top_left + top_right - 2 * left + 2 * right,
        -top_left - 2 * top - top_right + left + 2 * center + right,
    )
}

fn left_column_normal(img: ImageRef, y: u32) -> Normal {
    let top = img.alpha_at(0, y - 1);
    let top_right = img.alpha_at(1, y - 1);
    let center = img.alpha_at(0, y);
    let right = img.alpha_at(1, y);
    let bottom = img.alpha_at(0, y + 1);
    let bottom_right = img.alpha_at(1, y + 1);

    Normal::new(
        FACTOR_1_2,
        FACTOR_1_3,
        -top + top_right - 2 * center + 2 * right - bottom + bottom_right,
        -2 * top - top_right + 2 * bottom + bottom_right,
    )
}

fn right_column_normal(img: ImageRef, y: u32) -> Normal {
    let top_left = img.alpha_at(img.width - 2, y - 1);
    let top = img.alpha_at(img.width - 1, y - 1);
    let left = img.alpha_at(img.width - 2, y);
    let center = img.alpha_at(img.width - 1, y);
    let bottom_left = img.alpha_at(img.width - 2, y + 1);
    let bottom = img.alpha_at(img.width - 1, y + 1);

    Normal::new(
        FACTOR_1_2,
        FACTOR_1_3,
        -top_left + top - 2 * left + 2 * center - bottom_left + bottom,
        -top_left - 2 * top + bottom_left + 2 * bottom,
    )
}

fn interior_normal(img: ImageRef, x: u32, y: u32) -> Normal {
    let top_left = img.alpha_at(x - 1, y - 1);
    let top = img.alpha_at(x, y - 1);
    let top_right = img.alpha_at(x + 1, y - 1);
    let left = img.alpha_at(x - 1, y);
    let right = img.alpha_at(x + 1, y);
    let bottom_left = img.alpha_at(x - 1, y + 1);
    let bottom = img.alpha_at(x, y + 1);
    let bottom_right = img.alpha_at(x + 1, y + 1);

    Normal::new(
        FACTOR_1_4,
        FACTOR_1_4,
        -top_left + top_right - 2 * left + 2 * right - bottom_left + bottom_right,
        -top_left - 2 * top - top_right + bottom_left + 2 * bottom + bottom_right,
    )
}

fn calc_diffuse_alpha(_: u8, _: u8, _: u8) -> u8 {
    255
}

fn calc_specular_alpha(r: u8, g: u8, b: u8) -> u8 {
    use core::cmp::max;
    max(max(r, g), b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that the optimized specular lighting produces bit-exact results
    /// by comparing naive vs optimized implementations through the internal API.
    /// Since SpecularLighting fields are pub(crate) in usvg, we parse SVGs to
    /// create the filter, then extract and call both implementations.
    #[test]
    fn test_specular_optimized_vs_naive_bit_exact() {
        let sizes: &[(u32, u32)] = &[(4, 4), (16, 16), (64, 64), (100, 100)];
        let exponents: &[f32] = &[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 128.0];

        let light_configs: Vec<(&str, &str)> = vec![
            (
                "distant_45_45",
                r#"<feDistantLight azimuth="45" elevation="45"/>"#,
            ),
            (
                "distant_0_90",
                r#"<feDistantLight azimuth="0" elevation="90"/>"#,
            ),
            (
                "point_50_50_100",
                r#"<fePointLight x="50" y="50" z="100"/>"#,
            ),
            ("point_0_0_50", r#"<fePointLight x="0" y="0" z="50"/>"#),
            (
                "spot_basic",
                r#"<feSpotLight x="50" y="50" z="100" pointsAtX="128" pointsAtY="128" pointsAtZ="0"/>"#,
            ),
            (
                "spot_cone",
                r#"<feSpotLight x="25" y="25" z="200" pointsAtX="50" pointsAtY="50" pointsAtZ="0" specularExponent="10" limitingConeAngle="30"/>"#,
            ),
        ];

        fn make_test_image(width: u32, height: u32) -> Vec<RGBA8> {
            let len = (width * height) as usize;
            (0..len)
                .map(|i| {
                    let v = ((i * 37 + 13) % 256) as u8;
                    RGBA8 {
                        r: v,
                        g: v,
                        b: v,
                        a: v,
                    }
                })
                .collect()
        }

        for &(w, h) in sizes {
            let src_data = make_test_image(w, h);

            for &exp in exponents {
                for &(name, light_xml) in &light_configs {
                    let svg = format!(
                        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <filter id="f">
      <feSpecularLighting in="SourceGraphic" surfaceScale="1"
        specularConstant="1" specularExponent="{exp}"
        lighting-color="white" result="spec">
        {light_xml}
      </feSpecularLighting>
    </filter>
  </defs>
  <rect width="{w}" height="{h}" fill="gray" filter="url(#f)"/>
</svg>"##,
                    );

                    let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();

                    // Extract the SpecularLighting from the parsed tree.
                    // Walk the tree to find the filter with specular lighting.
                    let mut found = false;
                    for node in tree.root().children() {
                        if let usvg::Node::Group(group) = node {
                            for filter in group.filters() {
                                for primitive in filter.primitives() {
                                    if let usvg::filter::Kind::SpecularLighting(fe) =
                                        primitive.kind()
                                    {
                                        let light_source = fe.light_source();

                                        // Run naive
                                        let mut naive_dest = vec![
                                            RGBA8 {
                                                r: 0,
                                                g: 0,
                                                b: 0,
                                                a: 0
                                            };
                                            (w * h) as usize
                                        ];
                                        specular_lighting_naive(
                                            fe,
                                            light_source,
                                            ImageRef::new(w, h, &src_data),
                                            ImageRefMut::new(w, h, &mut naive_dest),
                                        );

                                        // Run optimized
                                        let mut opt_dest = vec![
                                            RGBA8 {
                                                r: 0,
                                                g: 0,
                                                b: 0,
                                                a: 0
                                            };
                                            (w * h) as usize
                                        ];
                                        specular_lighting_optimized(
                                            fe,
                                            light_source,
                                            ImageRef::new(w, h, &src_data),
                                            ImageRefMut::new(w, h, &mut opt_dest),
                                        );

                                        // Compare byte-for-byte
                                        for (idx, (n, o)) in
                                            naive_dest.iter().zip(opt_dest.iter()).enumerate()
                                        {
                                            assert!(
                                                n.r == o.r
                                                    && n.g == o.g
                                                    && n.b == o.b
                                                    && n.a == o.a,
                                                "Mismatch at pixel {} for {}x{} exp={} light={}: \
                                                 naive=({},{},{},{}) opt=({},{},{},{})",
                                                idx,
                                                w,
                                                h,
                                                exp,
                                                name,
                                                n.r,
                                                n.g,
                                                n.b,
                                                n.a,
                                                o.r,
                                                o.g,
                                                o.b,
                                                o.a,
                                            );
                                        }

                                        found = true;
                                    }
                                }
                            }
                        }
                    }
                    assert!(
                        found,
                        "Failed to find specular lighting for {}x{} exp={} light={}",
                        w, h, exp, name
                    );
                }
            }
        }
    }
}
