// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// An IIR blur.
//
// Based on http://www.getreuer.info/home/gaussianiir
//
// Licensed under 'Simplified BSD License'.
//
//
// Implements the fast Gaussian convolution algorithm of Alvarez and Mazorra,
// where the Gaussian is approximated by a cascade of first-order infinite
// impulsive response (IIR) filters.  Boundaries are handled with half-sample
// symmetric extension.
//
// Gaussian convolution is approached as approximating the heat equation and
// each timestep is performed with an efficient recursive computation.  Using
// more steps yields a more accurate approximation of the Gaussian.  A
// reasonable default value for `numsteps` is 4.
//
// Reference:
// Alvarez, Mazorra, "Signal and Image Restoration using Shock Filters and
// Anisotropic Diffusion," SIAM J. on Numerical Analysis, vol. 31, no. 2,
// pp. 590-605, 1994.

// TODO: Blurs right and bottom sides twice for some reason.

use super::ImageRefMut;
use rgb::ComponentSlice;

struct BlurData {
    width: usize,
    height: usize,
    sigma_x: f64,
    sigma_y: f64,
    steps: usize,
}

/// Applies an IIR blur.
///
/// Input image pixels should have a **premultiplied alpha**.
///
/// A negative or zero `sigma_x`/`sigma_y` will disable the blur along that axis.
///
/// # Allocations
///
/// This method will allocate a buffer for intermediate computation.
pub fn apply(sigma_x: f64, sigma_y: f64, src: ImageRefMut) {
    let pixel_count = (src.width as usize) * (src.height as usize);

    // For large images, the interleaved 4-channel implementation has better
    // cache locality and reduces 16 passes to 4. Use it as a cold early-return.
    if pixel_count > 500_000 {
        apply_interleaved(sigma_x, sigma_y, src);
        return;
    }

    // Default hot path: original per-channel implementation.
    let buf_size = pixel_count;
    let mut buf = vec![0.0; buf_size];
    let buf = &mut buf;

    let d = BlurData {
        width: src.width as usize,
        height: src.height as usize,
        sigma_x,
        sigma_y,
        steps: 4,
    };

    let data = src.data.as_mut_slice();
    gaussian_channel(data, &d, 0, buf);
    gaussian_channel(data, &d, 1, buf);
    gaussian_channel(data, &d, 2, buf);
    gaussian_channel(data, &d, 3, buf);
}

fn gaussian_channel(data: &mut [u8], d: &BlurData, channel: usize, buf: &mut [f64]) {
    for i in 0..data.len() / 4 {
        buf[i] = data[i * 4 + channel] as f64 / 255.0;
    }

    gaussianiir2d(d, buf);

    for i in 0..data.len() / 4 {
        data[i * 4 + channel] = (buf[i] * 255.0) as u8;
    }
}

fn gaussianiir2d(d: &BlurData, buf: &mut [f64]) {
    // Filter horizontally along each row.
    let (lambda_x, dnu_x) = if d.sigma_x > 0.0 {
        let (lambda, dnu) = gen_coefficients(d.sigma_x, d.steps);

        for y in 0..d.height {
            for _ in 0..d.steps {
                let idx = d.width * y;

                // Filter rightwards.
                for x in 1..d.width {
                    buf[idx + x] += dnu * buf[idx + x - 1];
                }

                let mut x = d.width - 1;

                // Filter leftwards.
                while x > 0 {
                    buf[idx + x - 1] += dnu * buf[idx + x];
                    x -= 1;
                }
            }
        }

        (lambda, dnu)
    } else {
        (1.0, 1.0)
    };

    // Filter vertically along each column.
    let (lambda_y, dnu_y) = if d.sigma_y > 0.0 {
        let (lambda, dnu) = gen_coefficients(d.sigma_y, d.steps);
        for x in 0..d.width {
            for _ in 0..d.steps {
                let idx = x;

                // Filter downwards.
                let mut y = d.width;
                while y < buf.len() {
                    buf[idx + y] += dnu * buf[idx + y - d.width];
                    y += d.width;
                }

                y = buf.len() - d.width;

                // Filter upwards.
                while y > 0 {
                    buf[idx + y - d.width] += dnu * buf[idx + y];
                    y -= d.width;
                }
            }
        }

        (lambda, dnu)
    } else {
        (1.0, 1.0)
    };

    let post_scale =
        ((dnu_x * dnu_y).sqrt() / (lambda_x * lambda_y).sqrt()).powi(2 * d.steps as i32);
    buf.iter_mut().for_each(|v| *v *= post_scale);
}

fn gen_coefficients(sigma: f64, steps: usize) -> (f64, f64) {
    let lambda = (sigma * sigma) / (2.0 * steps as f64);
    let dnu = (1.0 + 2.0 * lambda - (1.0 + 4.0 * lambda).sqrt()) / (2.0 * lambda);
    (lambda, dnu)
}

// ============================================================
// Optimized implementation (cold path for large images):
// all 4 channels interleaved, f64, tiled vertical
// ============================================================

/// Tile width for the vertical pass of the IIR filter.
const IIR_VERT_TILE_W: usize = 32;

/// Optimized IIR blur that processes all 4 RGBA channels simultaneously
/// using `[f64; 4]` arrays and tiles the vertical pass for better cache
/// locality. The `[f64; 4]` interleaved processing enables 4-channel SIMD
/// within a pixel (potential AVX 256-bit), but IIR has strict serial
/// dependency across pixels. The main win is reducing 16 passes to 4
/// passes (4 channels simultaneously instead of separately).
/// Uses f64 to stay bit-exact with the original implementation.
#[cold]
#[inline(never)]
fn apply_interleaved(sigma_x: f64, sigma_y: f64, src: ImageRefMut) {
    let width = src.width as usize;
    let height = src.height as usize;
    let pixel_count = width * height;
    let steps = 4usize;

    // Allocate interleaved f64 buffer: 4 channels per pixel.
    // This processes all channels in a single pass instead of 4 separate passes,
    // improving cache utilization by 4x.
    let mut buf = vec![[0.0f64; 4]; pixel_count];

    let data = src.data.as_mut_slice();

    // Convert u8 RGBA to f64 [0..1] interleaved.
    for i in 0..pixel_count {
        let base = i * 4;
        buf[i] = [
            data[base] as f64 / 255.0,
            data[base + 1] as f64 / 255.0,
            data[base + 2] as f64 / 255.0,
            data[base + 3] as f64 / 255.0,
        ];
    }

    // Filter horizontally along each row.
    let (lambda_x, dnu_x) = if sigma_x > 0.0 {
        let (lambda, dnu) = gen_coefficients(sigma_x, steps);

        for y in 0..height {
            let idx = width * y;

            for _ in 0..steps {
                // Filter rightwards.
                for x in 1..width {
                    let prev = buf[idx + x - 1];
                    let cur = &mut buf[idx + x];
                    cur[0] += dnu * prev[0];
                    cur[1] += dnu * prev[1];
                    cur[2] += dnu * prev[2];
                    cur[3] += dnu * prev[3];
                }

                // Filter leftwards.
                let mut x = width - 1;
                while x > 0 {
                    let next = buf[idx + x];
                    let cur = &mut buf[idx + x - 1];
                    cur[0] += dnu * next[0];
                    cur[1] += dnu * next[1];
                    cur[2] += dnu * next[2];
                    cur[3] += dnu * next[3];
                    x -= 1;
                }
            }
        }

        (lambda, dnu)
    } else {
        (1.0, 1.0)
    };

    // Filter vertically along each column, processing in tiles for cache locality.
    let (lambda_y, dnu_y) = if sigma_y > 0.0 {
        let (lambda, dnu) = gen_coefficients(sigma_y, steps);

        let mut col = 0;
        while col < width {
            let tile_end = std::cmp::min(col + IIR_VERT_TILE_W, width);

            for x in col..tile_end {
                for _ in 0..steps {
                    // Filter downwards.
                    let mut y_off = width;
                    while y_off < buf.len() {
                        let prev = buf[x + y_off - width];
                        let cur = &mut buf[x + y_off];
                        cur[0] += dnu * prev[0];
                        cur[1] += dnu * prev[1];
                        cur[2] += dnu * prev[2];
                        cur[3] += dnu * prev[3];
                        y_off += width;
                    }

                    // Filter upwards.
                    y_off = buf.len() - width;
                    while y_off > 0 {
                        let next = buf[x + y_off];
                        let cur = &mut buf[x + y_off - width];
                        cur[0] += dnu * next[0];
                        cur[1] += dnu * next[1];
                        cur[2] += dnu * next[2];
                        cur[3] += dnu * next[3];
                        y_off -= width;
                    }
                }
            }

            col = tile_end;
        }

        (lambda, dnu)
    } else {
        (1.0, 1.0)
    };

    // Apply post-scale and convert back to u8.
    let post_scale = ((dnu_x * dnu_y).sqrt() / (lambda_x * lambda_y).sqrt()).powi(2 * steps as i32);

    for i in 0..pixel_count {
        let base = i * 4;
        let px = buf[i];
        data[base] = (px[0] * post_scale * 255.0) as u8;
        data[base + 1] = (px[1] * post_scale * 255.0) as u8;
        data[base + 2] = (px[2] * post_scale * 255.0) as u8;
        data[base + 3] = (px[3] * post_scale * 255.0) as u8;
    }
}
