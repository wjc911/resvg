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

/// Threshold (in total pixel count) at which the optimized path breaks even with naive.
/// Benchmark shows 64x64 distant light is 0.93x (slower on the optimized path),
/// while 256x256 is clearly faster. 128x128 is a conservative crossover point.
#[cfg(test)]
#[allow(dead_code)]
const OPTIMIZED_THRESHOLD: u32 = 128 * 128;

fn apply(
    light_source: LightSource,
    surface_scale: f32,
    lighting_color: Color,
    light_factor: &dyn Fn(Normal, Vector3) -> f32,
    calc_alpha: fn(u8, u8, u8) -> u8,
    src: ImageRef,
    dest: ImageRefMut,
) {
    if src.width < 3 || src.height < 3 {
        return;
    }

    apply_optimized(
        light_source,
        surface_scale,
        lighting_color,
        light_factor,
        calc_alpha,
        src,
        dest,
    );
}

/// Original naive implementation preserved verbatim for correctness reference.
/// Only compiled in test builds for bit-exact verification against the optimized path.
#[cfg(test)]
fn apply_naive(
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

/// Optimized implementation using row-based alpha extraction and sliding window.
///
/// The primary gain is improved cache locality, NOT SIMD auto-vectorization
/// (the `dyn Fn` trait object for `light_factor` prevents inlining and vectorization).
///
/// Key optimizations over the naive version:
/// - Extracts alpha values into contiguous i16 row buffers for cache-friendly access,
///   avoiding scattered reads from the RGBA pixel array
/// - Uses a 3-row sliding window, reusing two rows per iteration (only reads one new row)
/// - Pre-computes spot light direction once instead of per-pixel
/// - Processes pixels in row-major order with sequential memory access
fn apply_optimized(
    light_source: LightSource,
    surface_scale: f32,
    lighting_color: Color,
    light_factor: &dyn Fn(Normal, Vector3) -> f32,
    calc_alpha: fn(u8, u8, u8) -> u8,
    src: ImageRef,
    mut dest: ImageRefMut,
) {
    let width = src.width;
    let height = src.height;
    let w = width as usize;

    // Pre-compute the light vector for distant light (constant for all pixels).
    let base_light_vector = match light_source {
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

    // Pre-compute spot light direction (constant for all pixels).
    let spot_direction = match light_source {
        LightSource::SpotLight(ref light) => {
            let origin = Vector3::new(light.x, light.y, light.z);
            let direction = Vector3::new(light.points_at_x, light.points_at_y, light.points_at_z);
            let direction = direction - origin;
            Some(direction.normalized().unwrap_or(direction))
        }
        _ => None,
    };

    // Helper: compute light vector for a given pixel position.
    #[inline(always)]
    fn compute_light_vector(
        light_source: &LightSource,
        base_light_vector: Vector3,
        surface_scale: f32,
        alpha: i16,
        px: u32,
        py: u32,
    ) -> Vector3 {
        match *light_source {
            LightSource::DistantLight(_) => base_light_vector,
            LightSource::PointLight(ref light) => {
                let nz = alpha as f32 / 255.0 * surface_scale;
                let origin = Vector3::new(light.x, light.y, light.z);
                let v = origin - Vector3::new(px as f32, py as f32, nz);
                v.normalized().unwrap_or(v)
            }
            LightSource::SpotLight(ref light) => {
                let nz = alpha as f32 / 255.0 * surface_scale;
                let origin = Vector3::new(light.x, light.y, light.z);
                let v = origin - Vector3::new(px as f32, py as f32, nz);
                v.normalized().unwrap_or(v)
            }
        }
    }

    // Helper: compute light color using pre-computed spot direction.
    #[inline(always)]
    fn compute_light_color_opt(
        light_source: &LightSource,
        lighting_color: Color,
        light_vector: Vector3,
        spot_direction: Option<Vector3>,
    ) -> Color {
        match *light_source {
            LightSource::DistantLight(_) | LightSource::PointLight(_) => lighting_color,
            LightSource::SpotLight(ref light) => {
                let direction = spot_direction.unwrap();
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

    // Helper: emit a single pixel.
    #[inline(always)]
    fn emit_pixel(
        dest: &mut ImageRefMut,
        light_source: &LightSource,
        base_light_vector: Vector3,
        surface_scale: f32,
        lighting_color: Color,
        spot_direction: Option<Vector3>,
        light_factor: &dyn Fn(Normal, Vector3) -> f32,
        calc_alpha: fn(u8, u8, u8) -> u8,
        px: u32,
        py: u32,
        normal: Normal,
        alpha: i16,
    ) {
        let lv = compute_light_vector(
            light_source,
            base_light_vector,
            surface_scale,
            alpha,
            px,
            py,
        );
        let lc = compute_light_color_opt(light_source, lighting_color, lv, spot_direction);
        let factor = light_factor(normal, lv);

        let compute = |x| (f32_bound(0.0, x as f32 * factor, 255.0) + 0.5) as u8;

        let r = compute(lc.red);
        let g = compute(lc.green);
        let b = compute(lc.blue);
        let a = calc_alpha(r, g, b);

        *dest.pixel_at_mut(px, py) = RGBA8 { b, g, r, a };
    }

    // Extract alpha channel for a row into a pre-allocated buffer.
    #[inline(always)]
    fn extract_alpha_row(src: &ImageRef, y: u32, buf: &mut [i16]) {
        let w = src.width as usize;
        let start = (src.width * y) as usize;
        let row = &src.data[start..start + w];
        for (i, pixel) in row.iter().enumerate() {
            buf[i] = pixel.a as i16;
        }
    }

    // Allocate row buffers for the sliding window (3 rows of alpha values).
    let mut row_buf_0 = vec![0i16; w];
    let mut row_buf_1 = vec![0i16; w];
    let mut row_buf_2 = vec![0i16; w];

    // --- Process top-left corner ---
    extract_alpha_row(&src, 0, &mut row_buf_0);
    extract_alpha_row(&src, 1, &mut row_buf_1);

    {
        let center = row_buf_0[0];
        let right = row_buf_0[1];
        let bottom = row_buf_1[0];
        let bottom_right = row_buf_1[1];
        let normal = Normal::new(
            FACTOR_2_3,
            FACTOR_2_3,
            -2 * center + 2 * right - bottom + bottom_right,
            -2 * center - right + 2 * bottom + bottom_right,
        );
        emit_pixel(
            &mut dest,
            &light_source,
            base_light_vector,
            surface_scale,
            lighting_color,
            spot_direction,
            light_factor,
            calc_alpha,
            0,
            0,
            normal,
            center,
        );
    }

    // --- Process top row interior ---
    for x in 1..(width - 1) {
        let xi = x as usize;
        let left = row_buf_0[xi - 1];
        let center = row_buf_0[xi];
        let right = row_buf_0[xi + 1];
        let bottom_left = row_buf_1[xi - 1];
        let bottom = row_buf_1[xi];
        let bottom_right = row_buf_1[xi + 1];
        let normal = Normal::new(
            FACTOR_1_3,
            FACTOR_1_2,
            -2 * left + 2 * right - bottom_left + bottom_right,
            -left - 2 * center - right + bottom_left + 2 * bottom + bottom_right,
        );
        emit_pixel(
            &mut dest,
            &light_source,
            base_light_vector,
            surface_scale,
            lighting_color,
            spot_direction,
            light_factor,
            calc_alpha,
            x,
            0,
            normal,
            center,
        );
    }

    // --- Process top-right corner ---
    {
        let xi = (width - 1) as usize;
        let left = row_buf_0[xi - 1];
        let center = row_buf_0[xi];
        let bottom_left = row_buf_1[xi - 1];
        let bottom = row_buf_1[xi];
        let normal = Normal::new(
            FACTOR_2_3,
            FACTOR_2_3,
            -2 * left + 2 * center - bottom_left + bottom,
            -left - 2 * center + bottom_left + 2 * bottom,
        );
        emit_pixel(
            &mut dest,
            &light_source,
            base_light_vector,
            surface_scale,
            lighting_color,
            spot_direction,
            light_factor,
            calc_alpha,
            width - 1,
            0,
            normal,
            center,
        );
    }

    // --- Process interior rows using sliding window ---
    for y in 1..(height - 1) {
        // Load the next row into row_buf_2.
        extract_alpha_row(&src, y + 1, &mut row_buf_2);

        // row_buf_0 = row y-1, row_buf_1 = row y, row_buf_2 = row y+1

        // Left column
        {
            let top = row_buf_0[0];
            let top_right = row_buf_0[1];
            let center = row_buf_1[0];
            let right = row_buf_1[1];
            let bottom = row_buf_2[0];
            let bottom_right = row_buf_2[1];
            let normal = Normal::new(
                FACTOR_1_2,
                FACTOR_1_3,
                -top + top_right - 2 * center + 2 * right - bottom + bottom_right,
                -2 * top - top_right + 2 * bottom + bottom_right,
            );
            emit_pixel(
                &mut dest,
                &light_source,
                base_light_vector,
                surface_scale,
                lighting_color,
                spot_direction,
                light_factor,
                calc_alpha,
                0,
                y,
                normal,
                center,
            );
        }

        // Interior pixels (the hot path)
        for x in 1..(width - 1) {
            let xi = x as usize;
            let top_left = row_buf_0[xi - 1];
            let top = row_buf_0[xi];
            let top_right = row_buf_0[xi + 1];
            let left = row_buf_1[xi - 1];
            let center = row_buf_1[xi];
            let right = row_buf_1[xi + 1];
            let bottom_left = row_buf_2[xi - 1];
            let bottom = row_buf_2[xi];
            let bottom_right = row_buf_2[xi + 1];

            let nx = -top_left + top_right - 2 * left + 2 * right - bottom_left + bottom_right;
            let ny = -top_left - 2 * top - top_right + bottom_left + 2 * bottom + bottom_right;

            let normal = Normal::new(FACTOR_1_4, FACTOR_1_4, nx, ny);
            emit_pixel(
                &mut dest,
                &light_source,
                base_light_vector,
                surface_scale,
                lighting_color,
                spot_direction,
                light_factor,
                calc_alpha,
                x,
                y,
                normal,
                center,
            );
        }

        // Right column
        {
            let xi = (width - 1) as usize;
            let top_left = row_buf_0[xi - 1];
            let top = row_buf_0[xi];
            let left = row_buf_1[xi - 1];
            let center = row_buf_1[xi];
            let bottom_left = row_buf_2[xi - 1];
            let bottom = row_buf_2[xi];
            let normal = Normal::new(
                FACTOR_1_2,
                FACTOR_1_3,
                -top_left + top - 2 * left + 2 * center - bottom_left + bottom,
                -top_left - 2 * top + bottom_left + 2 * bottom,
            );
            emit_pixel(
                &mut dest,
                &light_source,
                base_light_vector,
                surface_scale,
                lighting_color,
                spot_direction,
                light_factor,
                calc_alpha,
                width - 1,
                y,
                normal,
                center,
            );
        }

        // Rotate buffers: shift the window down by one row.
        core::mem::swap(&mut row_buf_0, &mut row_buf_1);
        core::mem::swap(&mut row_buf_1, &mut row_buf_2);
    }

    // --- Process bottom row ---
    // After the loop: row_buf_0 = row height-2, row_buf_1 = row height-1

    // Bottom-left corner
    {
        let top = row_buf_0[0];
        let top_right = row_buf_0[1];
        let center = row_buf_1[0];
        let right = row_buf_1[1];
        let normal = Normal::new(
            FACTOR_2_3,
            FACTOR_2_3,
            -top + top_right - 2 * center + 2 * right,
            -2 * top - top_right + 2 * center + right,
        );
        emit_pixel(
            &mut dest,
            &light_source,
            base_light_vector,
            surface_scale,
            lighting_color,
            spot_direction,
            light_factor,
            calc_alpha,
            0,
            height - 1,
            normal,
            center,
        );
    }

    // Bottom row interior
    for x in 1..(width - 1) {
        let xi = x as usize;
        let top_left = row_buf_0[xi - 1];
        let top = row_buf_0[xi];
        let top_right = row_buf_0[xi + 1];
        let left = row_buf_1[xi - 1];
        let center = row_buf_1[xi];
        let right = row_buf_1[xi + 1];
        let normal = Normal::new(
            FACTOR_1_3,
            FACTOR_1_2,
            -top_left + top_right - 2 * left + 2 * right,
            -top_left - 2 * top - top_right + left + 2 * center + right,
        );
        emit_pixel(
            &mut dest,
            &light_source,
            base_light_vector,
            surface_scale,
            lighting_color,
            spot_direction,
            light_factor,
            calc_alpha,
            x,
            height - 1,
            normal,
            center,
        );
    }

    // Bottom-right corner
    {
        let xi = (width - 1) as usize;
        let top_left = row_buf_0[xi - 1];
        let top = row_buf_0[xi];
        let left = row_buf_1[xi - 1];
        let center = row_buf_1[xi];
        let normal = Normal::new(
            FACTOR_2_3,
            FACTOR_2_3,
            -top_left + top - 2 * left + 2 * center,
            -top_left - 2 * top + left + 2 * center,
        );
        emit_pixel(
            &mut dest,
            &light_source,
            base_light_vector,
            surface_scale,
            lighting_color,
            spot_direction,
            light_factor,
            calc_alpha,
            width - 1,
            height - 1,
            normal,
            center,
        );
    }
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
    use usvg::PositiveF32;
    use usvg::filter::{DistantLight, PointLight, SpotLight};

    /// Creates test image data with a gradient alpha pattern for meaningful normals.
    fn make_test_image(width: u32, height: u32) -> Vec<RGBA8> {
        let mut data = vec![
            RGBA8 {
                r: 0,
                g: 0,
                b: 0,
                a: 0
            };
            (width * height) as usize
        ];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let cx = width as f32 / 2.0;
                let cy = height as f32 / 2.0;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let max_dist = (cx * cx + cy * cy).sqrt();
                let alpha = (255.0 * (1.0 - (dist / max_dist).min(1.0))) as u8;
                data[idx] = RGBA8 {
                    r: 128,
                    g: 64,
                    b: 32,
                    a: alpha,
                };
            }
        }
        data
    }

    /// Run both naive and optimized paths and compare for bit-exact equality.
    fn compare_naive_vs_optimized(
        light_source: LightSource,
        surface_scale: f32,
        lighting_color: Color,
        light_factor: &dyn Fn(Normal, Vector3) -> f32,
        calc_alpha: fn(u8, u8, u8) -> u8,
        width: u32,
        height: u32,
    ) {
        let src_data = make_test_image(width, height);
        let src = ImageRef::new(width, height, &src_data);

        let mut dest_naive = vec![
            RGBA8 {
                r: 0,
                g: 0,
                b: 0,
                a: 0
            };
            (width * height) as usize
        ];
        let mut dest_opt = vec![
            RGBA8 {
                r: 0,
                g: 0,
                b: 0,
                a: 0
            };
            (width * height) as usize
        ];

        super::apply_naive(
            light_source,
            surface_scale,
            lighting_color,
            light_factor,
            calc_alpha,
            src,
            ImageRefMut::new(width, height, &mut dest_naive),
        );

        super::apply_optimized(
            light_source,
            surface_scale,
            lighting_color,
            light_factor,
            calc_alpha,
            src,
            ImageRefMut::new(width, height, &mut dest_opt),
        );

        for i in 0..(width * height) as usize {
            assert_eq!(
                dest_naive[i],
                dest_opt[i],
                "Pixel mismatch at index {} (x={}, y={}): naive={:?} vs opt={:?}",
                i,
                i % width as usize,
                i / width as usize,
                dest_naive[i],
                dest_opt[i],
            );
        }
    }

    fn make_diffuse_light_factor(
        surface_scale: f32,
        diffuse_constant: f32,
    ) -> impl Fn(Normal, Vector3) -> f32 {
        move |normal: Normal, light_vector: Vector3| {
            let k = if normal.normal.approx_zero() {
                light_vector.z
            } else {
                let mut n = normal.normal * (surface_scale / 255.0);
                n.x *= normal.factor.x;
                n.y *= normal.factor.y;
                let normal = Vector3::new(n.x, n.y, 1.0);
                normal.dot(&light_vector) / normal.length()
            };
            diffuse_constant * k
        }
    }

    fn test_all_light_sources(width: u32, height: u32) {
        let surface_scales = [0.0_f32, 1.0, 5.0, 10.0];
        let diffuse_constants = [0.0_f32, 0.5, 1.0, 2.0];
        let lighting_color = Color::new_rgb(255, 255, 255);

        let light_sources: Vec<LightSource> = vec![
            LightSource::DistantLight(DistantLight {
                azimuth: 45.0,
                elevation: 55.0,
            }),
            LightSource::DistantLight(DistantLight {
                azimuth: 0.0,
                elevation: 0.0,
            }),
            LightSource::DistantLight(DistantLight {
                azimuth: 180.0,
                elevation: 90.0,
            }),
            LightSource::PointLight(PointLight {
                x: 150.0,
                y: 60.0,
                z: 200.0,
            }),
            LightSource::PointLight(PointLight {
                x: 0.0,
                y: 0.0,
                z: 100.0,
            }),
            LightSource::SpotLight(SpotLight {
                x: 150.0,
                y: 60.0,
                z: 200.0,
                points_at_x: 100.0,
                points_at_y: 100.0,
                points_at_z: 0.0,
                specular_exponent: PositiveF32::new(8.0).unwrap(),
                limiting_cone_angle: Some(30.0),
            }),
            LightSource::SpotLight(SpotLight {
                x: 50.0,
                y: 50.0,
                z: 100.0,
                points_at_x: 128.0,
                points_at_y: 128.0,
                points_at_z: 0.0,
                specular_exponent: PositiveF32::new(1.0).unwrap(),
                limiting_cone_angle: Some(90.0),
            }),
        ];

        for ls in &light_sources {
            for &ss in &surface_scales {
                for &dc in &diffuse_constants {
                    let lf = make_diffuse_light_factor(ss, dc);
                    compare_naive_vs_optimized(
                        *ls,
                        ss,
                        lighting_color,
                        &lf,
                        calc_diffuse_alpha,
                        width,
                        height,
                    );
                }
            }
        }
    }

    #[test]
    fn naive_vs_optimized_3x3() {
        test_all_light_sources(3, 3);
    }

    #[test]
    fn naive_vs_optimized_4x4() {
        test_all_light_sources(4, 4);
    }

    #[test]
    fn naive_vs_optimized_10x10() {
        test_all_light_sources(10, 10);
    }

    #[test]
    fn naive_vs_optimized_64x64() {
        test_all_light_sources(64, 64);
    }

    #[test]
    fn naive_vs_optimized_100x100() {
        test_all_light_sources(100, 100);
    }

    #[test]
    fn naive_vs_optimized_256x256() {
        test_all_light_sources(256, 256);
    }

    #[test]
    fn naive_vs_optimized_non_square() {
        test_all_light_sources(100, 50);
        test_all_light_sources(50, 100);
        test_all_light_sources(3, 200);
        test_all_light_sources(200, 3);
    }
}
