// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(clippy::needless_range_loop)]

use super::{ImageRefMut, f32_bound};
use usvg::ApproxZeroUlps;

const RAND_M: i32 = 2147483647; // 2**31 - 1
const RAND_A: i32 = 16807; // 7**5; primitive root of m
const RAND_Q: i32 = 127773; // m / a
const RAND_R: i32 = 2836; // m % a
const B_SIZE: usize = 0x100;
const B_SIZE_32: i32 = 0x100;
const B_LEN: usize = B_SIZE + B_SIZE + 2;
const BM: i32 = 0xff;
const PERLIN_N: i32 = 0x1000;

#[derive(Clone, Copy)]
struct StitchInfo {
    width: i32, // How much to subtract to wrap for stitching.
    height: i32,
    wrap_x: i32, // Minimum value to wrap.
    wrap_y: i32,
}

/// Applies a turbulence filter.
///
/// `dest` image pixels will have an **unpremultiplied alpha**.
///
/// - `offset_x` and `offset_y` indicate filter region offset.
/// - `sx` and `sy` indicate canvas scale.
pub fn apply(
    offset_x: f64,
    offset_y: f64,
    sx: f64,
    sy: f64,
    base_frequency_x: f64,
    base_frequency_y: f64,
    num_octaves: u32,
    seed: i32,
    stitch_tiles: bool,
    fractal_noise: bool,
    dest: ImageRefMut,
) {
    apply_optimized(
        offset_x,
        offset_y,
        sx,
        sy,
        base_frequency_x,
        base_frequency_y,
        num_octaves,
        seed,
        stitch_tiles,
        fractal_noise,
        dest,
    );
}

/// Original naive implementation preserved verbatim for correctness verification.
#[cfg(test)]
fn apply_naive(
    offset_x: f64,
    offset_y: f64,
    sx: f64,
    sy: f64,
    base_frequency_x: f64,
    base_frequency_y: f64,
    num_octaves: u32,
    seed: i32,
    stitch_tiles: bool,
    fractal_noise: bool,
    dest: ImageRefMut,
) {
    let (lattice_selector, gradient) = init_naive(seed);
    let width = dest.width;
    let height = dest.height;
    let mut x = 0;
    let mut y = 0;
    for pixel in dest.data.iter_mut() {
        let turb = |channel| {
            let (tx, ty) = ((x as f64 + offset_x) / sx, (y as f64 + offset_y) / sy);
            let n = turbulence_naive(
                channel,
                tx,
                ty,
                x as f64,
                y as f64,
                width as f64,
                height as f64,
                base_frequency_x,
                base_frequency_y,
                num_octaves,
                fractal_noise,
                stitch_tiles,
                &lattice_selector,
                &gradient,
            );

            let n = if fractal_noise {
                (n * 255.0 + 255.0) / 2.0
            } else {
                n * 255.0
            };

            (f32_bound(0.0, n as f32, 255.0) + 0.5) as u8
        };

        pixel.r = turb(0);
        pixel.g = turb(1);
        pixel.b = turb(2);
        pixel.a = turb(3);

        x += 1;
        if x == dest.width {
            x = 0;
            y += 1;
        }
    }
}

#[cfg(test)]
fn init_naive(mut seed: i32) -> (Vec<usize>, Vec<Vec<Vec<f64>>>) {
    let mut lattice_selector = vec![0; B_LEN];
    let mut gradient = vec![vec![vec![0.0; 2]; B_LEN]; 4];

    if seed <= 0 {
        seed = -seed % (RAND_M - 1) + 1;
    }

    if seed > RAND_M - 1 {
        seed = RAND_M - 1;
    }

    for k in 0..4 {
        for i in 0..B_SIZE {
            lattice_selector[i] = i;
            for j in 0..2 {
                seed = random(seed);
                gradient[k][i][j] =
                    ((seed % (B_SIZE_32 + B_SIZE_32)) - B_SIZE_32) as f64 / B_SIZE_32 as f64;
            }

            let s = (gradient[k][i][0] * gradient[k][i][0] + gradient[k][i][1] * gradient[k][i][1])
                .sqrt();

            gradient[k][i][0] /= s;
            gradient[k][i][1] /= s;
        }
    }

    for i in (1..B_SIZE).rev() {
        let k = lattice_selector[i];
        seed = random(seed);
        let j = (seed % B_SIZE_32) as usize;
        lattice_selector[i] = lattice_selector[j];
        lattice_selector[j] = k;
    }

    for i in 0..B_SIZE + 2 {
        lattice_selector[B_SIZE + i] = lattice_selector[i];
        for g in gradient.iter_mut().take(4) {
            for j in 0..2 {
                g[B_SIZE + i][j] = g[i][j];
            }
        }
    }

    (lattice_selector, gradient)
}

#[cfg(test)]
fn turbulence_naive(
    color_channel: usize,
    mut x: f64,
    mut y: f64,
    tile_x: f64,
    tile_y: f64,
    tile_width: f64,
    tile_height: f64,
    mut base_freq_x: f64,
    mut base_freq_y: f64,
    num_octaves: u32,
    fractal_sum: bool,
    do_stitching: bool,
    lattice_selector: &[usize],
    gradient: &[Vec<Vec<f64>>],
) -> f64 {
    // Adjust the base frequencies if necessary for stitching.
    let mut stitch = if do_stitching {
        // When stitching tiled turbulence, the frequencies must be adjusted
        // so that the tile borders will be continuous.
        if !base_freq_x.approx_zero_ulps(4) {
            let lo_freq = (tile_width * base_freq_x).floor() / tile_width;
            let hi_freq = (tile_width * base_freq_x).ceil() / tile_width;
            if base_freq_x / lo_freq < hi_freq / base_freq_x {
                base_freq_x = lo_freq;
            } else {
                base_freq_x = hi_freq;
            }
        }

        if !base_freq_y.approx_zero_ulps(4) {
            let lo_freq = (tile_height * base_freq_y).floor() / tile_height;
            let hi_freq = (tile_height * base_freq_y).ceil() / tile_height;
            if base_freq_y / lo_freq < hi_freq / base_freq_y {
                base_freq_y = lo_freq;
            } else {
                base_freq_y = hi_freq;
            }
        }

        // Set up initial stitch values.
        let width = (tile_width * base_freq_x + 0.5) as i32;
        let height = (tile_height * base_freq_y + 0.5) as i32;
        let wrap_x = (tile_x * base_freq_x + PERLIN_N as f64 + width as f64) as i32;
        let wrap_y = (tile_y * base_freq_y + PERLIN_N as f64 + height as f64) as i32;
        Some(StitchInfo {
            width,
            height,
            wrap_x,
            wrap_y,
        })
    } else {
        None
    };

    let mut sum = 0.0;
    x *= base_freq_x;
    y *= base_freq_y;
    let mut ratio = 1.0;
    for _ in 0..num_octaves {
        if fractal_sum {
            sum += noise2_naive(color_channel, x, y, lattice_selector, gradient, stitch) / ratio;
        } else {
            sum +=
                noise2_naive(color_channel, x, y, lattice_selector, gradient, stitch).abs() / ratio;
        }
        x *= 2.0;
        y *= 2.0;
        ratio *= 2.0;

        if let Some(ref mut stitch) = stitch {
            // Update stitch values. Subtracting PerlinN before the multiplication and
            // adding it afterward simplifies to subtracting it once.
            stitch.width *= 2;
            stitch.wrap_x = 2 * stitch.wrap_x - PERLIN_N;
            stitch.height *= 2;
            stitch.wrap_y = 2 * stitch.wrap_y - PERLIN_N;
        }
    }

    sum
}

#[cfg(test)]
fn noise2_naive(
    color_channel: usize,
    x: f64,
    y: f64,
    lattice_selector: &[usize],
    gradient: &[Vec<Vec<f64>>],
    stitch_info: Option<StitchInfo>,
) -> f64 {
    let t = x + PERLIN_N as f64;
    let mut bx0 = t as i32;
    let mut bx1 = bx0 + 1;
    let rx0 = t - t as i64 as f64;
    let rx1 = rx0 - 1.0;
    let t = y + PERLIN_N as f64;
    let mut by0 = t as i32;
    let mut by1 = by0 + 1;
    let ry0 = t - t as i64 as f64;
    let ry1 = ry0 - 1.0;

    // If stitching, adjust lattice points accordingly.
    if let Some(info) = stitch_info {
        if bx0 >= info.wrap_x {
            bx0 -= info.width;
        }

        if bx1 >= info.wrap_x {
            bx1 -= info.width;
        }

        if by0 >= info.wrap_y {
            by0 -= info.height;
        }

        if by1 >= info.wrap_y {
            by1 -= info.height;
        }
    }

    bx0 &= BM;
    bx1 &= BM;
    by0 &= BM;
    by1 &= BM;
    let i = lattice_selector[bx0 as usize];
    let j = lattice_selector[bx1 as usize];
    let b00 = lattice_selector[i + by0 as usize];
    let b10 = lattice_selector[j + by0 as usize];
    let b01 = lattice_selector[i + by1 as usize];
    let b11 = lattice_selector[j + by1 as usize];
    let sx = s_curve(rx0);
    let sy = s_curve(ry0);
    let q = &gradient[color_channel][b00];
    let u = rx0 * q[0] + ry0 * q[1];
    let q = &gradient[color_channel][b10];
    let v = rx1 * q[0] + ry0 * q[1];
    let a = lerp(sx, u, v);
    let q = &gradient[color_channel][b01];
    let u = rx0 * q[0] + ry1 * q[1];
    let q = &gradient[color_channel][b11];
    let v = rx1 * q[0] + ry1 * q[1];
    let b = lerp(sx, u, v);
    lerp(sy, a, b)
}

// ---------------------------------------------------------------------------
// Optimized implementation
// ---------------------------------------------------------------------------

/// Flat gradient table: gradient_flat[channel][index] = [gx, gy]
/// Stored as fixed-size arrays for cache-friendly access (no heap indirection).
struct GradientTable {
    /// gradient[channel][lattice_index] = (gx, gy)
    data: [[[f64; 2]; B_LEN]; 4],
}

/// Optimized init: produces flat arrays instead of nested Vecs.
fn init_optimized(mut seed: i32) -> ([usize; B_LEN], GradientTable) {
    let mut lattice_selector = [0usize; B_LEN];
    let mut gradient = GradientTable {
        data: [[[0.0; 2]; B_LEN]; 4],
    };

    if seed <= 0 {
        seed = -seed % (RAND_M - 1) + 1;
    }

    if seed > RAND_M - 1 {
        seed = RAND_M - 1;
    }

    for k in 0..4 {
        for i in 0..B_SIZE {
            lattice_selector[i] = i;
            for j in 0..2 {
                seed = random(seed);
                gradient.data[k][i][j] =
                    ((seed % (B_SIZE_32 + B_SIZE_32)) - B_SIZE_32) as f64 / B_SIZE_32 as f64;
            }

            let s = (gradient.data[k][i][0] * gradient.data[k][i][0]
                + gradient.data[k][i][1] * gradient.data[k][i][1])
                .sqrt();

            gradient.data[k][i][0] /= s;
            gradient.data[k][i][1] /= s;
        }
    }

    for i in (1..B_SIZE).rev() {
        let k = lattice_selector[i];
        seed = random(seed);
        let j = (seed % B_SIZE_32) as usize;
        lattice_selector[i] = lattice_selector[j];
        lattice_selector[j] = k;
    }

    for i in 0..B_SIZE + 2 {
        lattice_selector[B_SIZE + i] = lattice_selector[i];
        for k in 0..4 {
            for j in 0..2 {
                gradient.data[k][B_SIZE + i][j] = gradient.data[k][i][j];
            }
        }
    }

    (lattice_selector, gradient)
}

/// Compute noise for all 4 channels at once, sharing lattice lookups.
/// Returns [channel0, channel1, channel2, channel3].
#[inline]
fn noise2_4ch(
    x: f64,
    y: f64,
    lattice_selector: &[usize; B_LEN],
    gradient: &GradientTable,
    stitch_info: Option<StitchInfo>,
) -> [f64; 4] {
    let t = x + PERLIN_N as f64;
    let mut bx0 = t as i32;
    let mut bx1 = bx0 + 1;
    let rx0 = t - t as i64 as f64;
    let rx1 = rx0 - 1.0;
    let t = y + PERLIN_N as f64;
    let mut by0 = t as i32;
    let mut by1 = by0 + 1;
    let ry0 = t - t as i64 as f64;
    let ry1 = ry0 - 1.0;

    // If stitching, adjust lattice points accordingly.
    if let Some(info) = stitch_info {
        if bx0 >= info.wrap_x {
            bx0 -= info.width;
        }
        if bx1 >= info.wrap_x {
            bx1 -= info.width;
        }
        if by0 >= info.wrap_y {
            by0 -= info.height;
        }
        if by1 >= info.wrap_y {
            by1 -= info.height;
        }
    }

    bx0 &= BM;
    bx1 &= BM;
    by0 &= BM;
    by1 &= BM;

    let i = lattice_selector[bx0 as usize];
    let j = lattice_selector[bx1 as usize];
    let b00 = lattice_selector[i + by0 as usize];
    let b10 = lattice_selector[j + by0 as usize];
    let b01 = lattice_selector[i + by1 as usize];
    let b11 = lattice_selector[j + by1 as usize];
    let sx = s_curve(rx0);
    let sy = s_curve(ry0);

    // Process all 4 channels using the shared lattice points
    let mut result = [0.0f64; 4];
    for ch in 0..4 {
        let g = &gradient.data[ch];
        let q00 = &g[b00];
        let u = rx0 * q00[0] + ry0 * q00[1];
        let q10 = &g[b10];
        let v = rx1 * q10[0] + ry0 * q10[1];
        let a = lerp(sx, u, v);
        let q01 = &g[b01];
        let u = rx0 * q01[0] + ry1 * q01[1];
        let q11 = &g[b11];
        let v = rx1 * q11[0] + ry1 * q11[1];
        let b = lerp(sx, u, v);
        result[ch] = lerp(sy, a, b);
    }
    result
}

/// Compute turbulence for all 4 channels at once for a single pixel.
/// Returns [channel0, channel1, channel2, channel3].
#[inline]
fn turbulence_4ch(
    mut x: f64,
    mut y: f64,
    base_freq_x: f64,
    base_freq_y: f64,
    num_octaves: u32,
    fractal_sum: bool,
    mut stitch: Option<StitchInfo>,
    lattice_selector: &[usize; B_LEN],
    gradient: &GradientTable,
) -> [f64; 4] {
    let mut sums = [0.0f64; 4];
    x *= base_freq_x;
    y *= base_freq_y;
    let mut inv_ratio = 1.0_f64;
    for _ in 0..num_octaves {
        let n = noise2_4ch(x, y, lattice_selector, gradient, stitch);
        if fractal_sum {
            for ch in 0..4 {
                sums[ch] += n[ch] * inv_ratio;
            }
        } else {
            for ch in 0..4 {
                sums[ch] += n[ch].abs() * inv_ratio;
            }
        }
        x *= 2.0;
        y *= 2.0;
        inv_ratio *= 0.5;

        if let Some(ref mut s) = stitch {
            s.width *= 2;
            s.wrap_x = 2 * s.wrap_x - PERLIN_N;
            s.height *= 2;
            s.wrap_y = 2 * s.wrap_y - PERLIN_N;
        }
    }
    sums
}

fn apply_optimized(
    offset_x: f64,
    offset_y: f64,
    sx: f64,
    sy: f64,
    base_frequency_x: f64,
    base_frequency_y: f64,
    num_octaves: u32,
    seed: i32,
    stitch_tiles: bool,
    fractal_noise: bool,
    dest: ImageRefMut,
) {
    let (lattice_selector, gradient) = init_optimized(seed);
    let width = dest.width;
    let height = dest.height;

    // Pre-compute adjusted base frequencies and initial stitch info (once, not per pixel).
    let mut base_freq_x = base_frequency_x;
    let mut base_freq_y = base_frequency_y;
    let tile_width = width as f64;
    let tile_height = height as f64;

    let initial_stitch = if stitch_tiles {
        if !base_freq_x.approx_zero_ulps(4) {
            let lo_freq = (tile_width * base_freq_x).floor() / tile_width;
            let hi_freq = (tile_width * base_freq_x).ceil() / tile_width;
            if base_freq_x / lo_freq < hi_freq / base_freq_x {
                base_freq_x = lo_freq;
            } else {
                base_freq_x = hi_freq;
            }
        }

        if !base_freq_y.approx_zero_ulps(4) {
            let lo_freq = (tile_height * base_freq_y).floor() / tile_height;
            let hi_freq = (tile_height * base_freq_y).ceil() / tile_height;
            if base_freq_y / lo_freq < hi_freq / base_freq_y {
                base_freq_y = lo_freq;
            } else {
                base_freq_y = hi_freq;
            }
        }

        // sw and sh depend only on tile dimensions and base frequencies — constant per frame.
        let sw = (tile_width * base_freq_x + 0.5) as i32;
        let sh = (tile_height * base_freq_y + 0.5) as i32;
        Some((base_freq_x, base_freq_y, sw, sh))
    } else {
        None
    };

    let mut px = 0u32;
    let mut py = 0u32;
    for pixel in dest.data.iter_mut() {
        let tx = (px as f64 + offset_x) / sx;
        let ty = (py as f64 + offset_y) / sy;

        // Build per-pixel stitch info; only wrap_x/wrap_y vary per pixel.
        let stitch = initial_stitch.map(|(bfx, bfy, sw, sh)| {
            let wx = (px as f64 * bfx + PERLIN_N as f64 + sw as f64) as i32;
            let wy = (py as f64 * bfy + PERLIN_N as f64 + sh as f64) as i32;
            StitchInfo {
                width: sw,
                height: sh,
                wrap_x: wx,
                wrap_y: wy,
            }
        });

        let n = turbulence_4ch(
            tx,
            ty,
            base_freq_x,
            base_freq_y,
            num_octaves,
            fractal_noise,
            stitch,
            &lattice_selector,
            &gradient,
        );

        fn to_u8(val: f64, fractal_noise: bool) -> u8 {
            let v = if fractal_noise {
                (val * 255.0 + 255.0) / 2.0
            } else {
                val * 255.0
            };
            (f32_bound(0.0, v as f32, 255.0) + 0.5) as u8
        }

        pixel.r = to_u8(n[0], fractal_noise);
        pixel.g = to_u8(n[1], fractal_noise);
        pixel.b = to_u8(n[2], fractal_noise);
        pixel.a = to_u8(n[3], fractal_noise);

        px += 1;
        if px == width {
            px = 0;
            py += 1;
        }
    }
}

fn random(seed: i32) -> i32 {
    let mut result = RAND_A * (seed % RAND_Q) - RAND_R * (seed / RAND_Q);
    if result <= 0 {
        result += RAND_M;
    }

    result
}

#[inline]
fn s_curve(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rgb::RGBA8;

    fn make_image(w: u32, h: u32) -> Vec<RGBA8> {
        vec![
            RGBA8 {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            };
            (w * h) as usize
        ]
    }

    /// Exhaustive bit-exact correctness test across all parameter combinations.
    #[test]
    fn test_bit_exact_correctness() {
        let resolutions = [(16, 16), (64, 64), (100, 73), (256, 256)];
        let octaves_list = [1, 2, 4, 8];
        let seeds = [0, 1, 42, -7, 2147483646];
        let freq_pairs = [
            (0.01, 0.01),
            (0.05, 0.05),
            (0.1, 0.02),
            (0.0, 0.05),
            (0.05, 0.0),
        ];

        for &(w, h) in &resolutions {
            for &num_octaves in &octaves_list {
                for &seed in &seeds {
                    for &(bfx, bfy) in &freq_pairs {
                        for &fractal_noise in &[true, false] {
                            for &stitch_tiles in &[true, false] {
                                for &(offset_x, offset_y) in &[(0.0, 0.0), (10.5, -3.2)] {
                                    let sx = 1.0;
                                    let sy = 1.0;

                                    let mut naive_buf = make_image(w, h);
                                    let mut opt_buf = make_image(w, h);

                                    apply_naive(
                                        offset_x,
                                        offset_y,
                                        sx,
                                        sy,
                                        bfx,
                                        bfy,
                                        num_octaves,
                                        seed,
                                        stitch_tiles,
                                        fractal_noise,
                                        ImageRefMut::new(w, h, &mut naive_buf),
                                    );

                                    apply_optimized(
                                        offset_x,
                                        offset_y,
                                        sx,
                                        sy,
                                        bfx,
                                        bfy,
                                        num_octaves,
                                        seed,
                                        stitch_tiles,
                                        fractal_noise,
                                        ImageRefMut::new(w, h, &mut opt_buf),
                                    );

                                    for (idx, (n, o)) in
                                        naive_buf.iter().zip(opt_buf.iter()).enumerate()
                                    {
                                        assert_eq!(
                                            *n,
                                            *o,
                                            "Mismatch at pixel {} for {}x{} octaves={} seed={} \
                                             freq=({},{}) fractal={} stitch={} offset=({},{}): \
                                             naive={:?} opt={:?}",
                                            idx,
                                            w,
                                            h,
                                            num_octaves,
                                            seed,
                                            bfx,
                                            bfy,
                                            fractal_noise,
                                            stitch_tiles,
                                            offset_x,
                                            offset_y,
                                            n,
                                            o
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Performance comparison test (run with --release --nocapture to see output).
    #[test]
    fn test_performance_comparison() {
        use std::time::Instant;

        let configs: &[(u32, u32, u32, bool, bool, &str)] = &[
            (256, 256, 1, false, false, "256x256 oct=1 turb"),
            (256, 256, 4, false, false, "256x256 oct=4 turb"),
            (256, 256, 8, false, false, "256x256 oct=8 turb"),
            (256, 256, 1, false, true, "256x256 oct=1 turb stitch"),
            (256, 256, 4, false, true, "256x256 oct=4 turb stitch"),
            (256, 256, 1, true, false, "256x256 oct=1 fractal"),
            (256, 256, 4, true, false, "256x256 oct=4 fractal"),
            (1024, 1024, 1, false, false, "1024x1024 oct=1 turb"),
            (1024, 1024, 4, false, false, "1024x1024 oct=4 turb"),
            (1024, 1024, 8, false, false, "1024x1024 oct=8 turb"),
        ];

        println!(
            "\n{:<30} {:>12} {:>12} {:>8}",
            "Config", "Naive (ms)", "Opt (ms)", "Speedup"
        );
        println!("{}", "-".repeat(65));

        for &(w, h, octaves, fractal, stitch, label) in configs {
            let iters = if w >= 1024 { 3 } else { 10 };

            // Warm up naive
            {
                let mut buf = make_image(w, h);
                apply_naive(
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.05,
                    0.05,
                    octaves,
                    42,
                    stitch,
                    fractal,
                    ImageRefMut::new(w, h, &mut buf),
                );
            }
            let start = Instant::now();
            for _ in 0..iters {
                let mut buf = make_image(w, h);
                apply_naive(
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.05,
                    0.05,
                    octaves,
                    42,
                    stitch,
                    fractal,
                    ImageRefMut::new(w, h, &mut buf),
                );
                std::hint::black_box(&buf);
            }
            let naive_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // Warm up optimized
            {
                let mut buf = make_image(w, h);
                apply_optimized(
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.05,
                    0.05,
                    octaves,
                    42,
                    stitch,
                    fractal,
                    ImageRefMut::new(w, h, &mut buf),
                );
            }
            let start = Instant::now();
            for _ in 0..iters {
                let mut buf = make_image(w, h);
                apply_optimized(
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.05,
                    0.05,
                    octaves,
                    42,
                    stitch,
                    fractal,
                    ImageRefMut::new(w, h, &mut buf),
                );
                std::hint::black_box(&buf);
            }
            let opt_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let speedup = naive_ms / opt_ms;
            println!(
                "{:<30} {:>12.3} {:>12.3} {:>7.2}x",
                label, naive_ms, opt_ms, speedup
            );
        }
        println!();
    }

    /// Test with non-unit scale factors.
    #[test]
    fn test_bit_exact_with_scale() {
        let w = 64;
        let h = 64;
        let scales = [(1.0, 1.0), (2.0, 2.0), (0.5, 1.5), (3.0, 0.75)];

        for &(sx, sy) in &scales {
            for &fractal_noise in &[true, false] {
                for &stitch_tiles in &[true, false] {
                    let mut naive_buf = make_image(w, h);
                    let mut opt_buf = make_image(w, h);

                    apply_naive(
                        5.0,
                        -2.0,
                        sx,
                        sy,
                        0.05,
                        0.05,
                        4,
                        42,
                        stitch_tiles,
                        fractal_noise,
                        ImageRefMut::new(w, h, &mut naive_buf),
                    );

                    apply_optimized(
                        5.0,
                        -2.0,
                        sx,
                        sy,
                        0.05,
                        0.05,
                        4,
                        42,
                        stitch_tiles,
                        fractal_noise,
                        ImageRefMut::new(w, h, &mut opt_buf),
                    );

                    assert_eq!(
                        naive_buf, opt_buf,
                        "Mismatch for scale=({},{}) fractal={} stitch={}",
                        sx, sy, fractal_noise, stitch_tiles
                    );
                }
            }
        }
    }
}
