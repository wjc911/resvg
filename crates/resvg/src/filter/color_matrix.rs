// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{ImageRefMut, f32_bound};
use rgb::RGBA8;
use usvg::filter::ColorMatrixKind as ColorMatrix;

/// Applies a color matrix filter.
///
/// Input image pixels should have an **unpremultiplied alpha**.
///
/// For the full 4x5 `Matrix` variant the row-major matrix is transposed into
/// five column vectors of `[f32; 4]` so that LLVM can auto-vectorize the
/// per-pixel matrix multiply into packed SIMD instructions.
///
/// The `Saturate`, `HueRotate`, and `LuminanceToAlpha` variants use compact
/// specialised loops that the compiler already handles well.
pub fn apply(matrix: &ColorMatrix, src: ImageRefMut) {
    match matrix {
        ColorMatrix::Matrix(m) => {
            // Transpose the 4x5 row-major matrix into 5 column vectors of length 4.
            // Row-major layout: row i = m[i*5 .. i*5+5]
            // Column j holds [m[0*5+j], m[1*5+j], m[2*5+j], m[3*5+j]]
            let col0 = [m[0], m[5], m[10], m[15]]; // R column
            let col1 = [m[1], m[6], m[11], m[16]]; // G column
            let col2 = [m[2], m[7], m[12], m[17]]; // B column
            let col3 = [m[3], m[8], m[13], m[18]]; // A column
            let col4 = [m[4], m[9], m[14], m[19]]; // bias column

            apply_matrix_cols(src, &col0, &col1, &col2, &col3, &col4);
        }
        ColorMatrix::Saturate(v) => {
            let v = v.get().max(0.0);
            let m = [
                0.213 + 0.787 * v,
                0.715 - 0.715 * v,
                0.072 - 0.072 * v,
                0.213 - 0.213 * v,
                0.715 + 0.285 * v,
                0.072 - 0.072 * v,
                0.213 - 0.213 * v,
                0.715 - 0.715 * v,
                0.072 + 0.928 * v,
            ];

            for pixel in src.data {
                let (r, g, b, _) = to_normalized_components(*pixel);

                let new_r = r * m[0] + g * m[1] + b * m[2];
                let new_g = r * m[3] + g * m[4] + b * m[5];
                let new_b = r * m[6] + g * m[7] + b * m[8];

                pixel.r = from_normalized(new_r);
                pixel.g = from_normalized(new_g);
                pixel.b = from_normalized(new_b);
            }
        }
        ColorMatrix::HueRotate(angle) => {
            let angle = angle.to_radians();
            let a1 = angle.cos();
            let a2 = angle.sin();
            let m = [
                0.213 + 0.787 * a1 - 0.213 * a2,
                0.715 - 0.715 * a1 - 0.715 * a2,
                0.072 - 0.072 * a1 + 0.928 * a2,
                0.213 - 0.213 * a1 + 0.143 * a2,
                0.715 + 0.285 * a1 + 0.140 * a2,
                0.072 - 0.072 * a1 - 0.283 * a2,
                0.213 - 0.213 * a1 - 0.787 * a2,
                0.715 - 0.715 * a1 + 0.715 * a2,
                0.072 + 0.928 * a1 + 0.072 * a2,
            ];

            for pixel in src.data {
                let (r, g, b, _) = to_normalized_components(*pixel);

                let new_r = r * m[0] + g * m[1] + b * m[2];
                let new_g = r * m[3] + g * m[4] + b * m[5];
                let new_b = r * m[6] + g * m[7] + b * m[8];

                pixel.r = from_normalized(new_r);
                pixel.g = from_normalized(new_g);
                pixel.b = from_normalized(new_b);
            }
        }
        ColorMatrix::LuminanceToAlpha => {
            for pixel in src.data {
                let (r, g, b, _) = to_normalized_components(*pixel);

                let new_a = r * 0.2125 + g * 0.7154 + b * 0.0721;

                pixel.r = 0;
                pixel.g = 0;
                pixel.b = 0;
                pixel.a = from_normalized(new_a);
            }
        }
    }
}

/// Optimised full 4x5 matrix apply using column-major `[f32; 4]` vectors.
///
/// The matrix is expressed as 5 column vectors (col0..col4), each `[f32; 4]`.
/// For each pixel the accumulation order matches the naive row-major order:
///   `out[i] = ((((col0[i]*r) + (col1[i]*g)) + (col2[i]*b)) + (col3[i]*a)) + col4[i]`
///
/// This exact left-to-right order guarantees bit-exact results with the
/// original scalar implementation while allowing LLVM to auto-vectorize the
/// 4-wide operations into packed SIMD instructions (all 4 lanes execute the
/// same sequence of operations on independent data).
#[inline(never)]
fn apply_matrix_cols(
    src: ImageRefMut,
    col0: &[f32; 4],
    col1: &[f32; 4],
    col2: &[f32; 4],
    col3: &[f32; 4],
    col4: &[f32; 4],
) {
    for pixel in src.data {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;
        let a = pixel.a as f32 / 255.0;

        // Accumulate in the exact same left-to-right order as the naive
        // implementation: r*coeff + g*coeff + b*coeff + a*coeff + bias.
        // The `[f32; 4]` indexing with a fixed 0..4 range lets LLVM
        // auto-vectorize all 4 lanes into packed SIMD ops.
        let mut out = [0.0_f32; 4];
        for i in 0..4 {
            out[i] = col0[i] * r;
            out[i] += col1[i] * g;
            out[i] += col2[i] * b;
            out[i] += col3[i] * a;
            out[i] += col4[i];
        }

        pixel.r = from_normalized(out[0]);
        pixel.g = from_normalized(out[1]);
        pixel.b = from_normalized(out[2]);
        pixel.a = from_normalized(out[3]);
    }
}

/// Verbatim copy of the original naive implementation, preserved for
/// correctness testing and benchmarking.
#[allow(dead_code)]
pub fn apply_naive(matrix: &ColorMatrix, src: ImageRefMut) {
    match matrix {
        ColorMatrix::Matrix(m) => {
            for pixel in src.data {
                let (r, g, b, a) = to_normalized_components(*pixel);

                let new_r = r * m[0] + g * m[1] + b * m[2] + a * m[3] + m[4];
                let new_g = r * m[5] + g * m[6] + b * m[7] + a * m[8] + m[9];
                let new_b = r * m[10] + g * m[11] + b * m[12] + a * m[13] + m[14];
                let new_a = r * m[15] + g * m[16] + b * m[17] + a * m[18] + m[19];

                pixel.r = from_normalized(new_r);
                pixel.g = from_normalized(new_g);
                pixel.b = from_normalized(new_b);
                pixel.a = from_normalized(new_a);
            }
        }
        ColorMatrix::Saturate(v) => {
            let v = v.get().max(0.0);
            let m = [
                0.213 + 0.787 * v,
                0.715 - 0.715 * v,
                0.072 - 0.072 * v,
                0.213 - 0.213 * v,
                0.715 + 0.285 * v,
                0.072 - 0.072 * v,
                0.213 - 0.213 * v,
                0.715 - 0.715 * v,
                0.072 + 0.928 * v,
            ];

            for pixel in src.data {
                let (r, g, b, _) = to_normalized_components(*pixel);

                let new_r = r * m[0] + g * m[1] + b * m[2];
                let new_g = r * m[3] + g * m[4] + b * m[5];
                let new_b = r * m[6] + g * m[7] + b * m[8];

                pixel.r = from_normalized(new_r);
                pixel.g = from_normalized(new_g);
                pixel.b = from_normalized(new_b);
            }
        }
        ColorMatrix::HueRotate(angle) => {
            let angle = angle.to_radians();
            let a1 = angle.cos();
            let a2 = angle.sin();
            let m = [
                0.213 + 0.787 * a1 - 0.213 * a2,
                0.715 - 0.715 * a1 - 0.715 * a2,
                0.072 - 0.072 * a1 + 0.928 * a2,
                0.213 - 0.213 * a1 + 0.143 * a2,
                0.715 + 0.285 * a1 + 0.140 * a2,
                0.072 - 0.072 * a1 - 0.283 * a2,
                0.213 - 0.213 * a1 - 0.787 * a2,
                0.715 - 0.715 * a1 + 0.715 * a2,
                0.072 + 0.928 * a1 + 0.072 * a2,
            ];

            for pixel in src.data {
                let (r, g, b, _) = to_normalized_components(*pixel);

                let new_r = r * m[0] + g * m[1] + b * m[2];
                let new_g = r * m[3] + g * m[4] + b * m[5];
                let new_b = r * m[6] + g * m[7] + b * m[8];

                pixel.r = from_normalized(new_r);
                pixel.g = from_normalized(new_g);
                pixel.b = from_normalized(new_b);
            }
        }
        ColorMatrix::LuminanceToAlpha => {
            for pixel in src.data {
                let (r, g, b, _) = to_normalized_components(*pixel);

                let new_a = r * 0.2125 + g * 0.7154 + b * 0.0721;

                pixel.r = 0;
                pixel.g = 0;
                pixel.b = 0;
                pixel.a = from_normalized(new_a);
            }
        }
    }
}

#[inline]
fn to_normalized_components(pixel: RGBA8) -> (f32, f32, f32, f32) {
    (
        pixel.r as f32 / 255.0,
        pixel.g as f32 / 255.0,
        pixel.b as f32 / 255.0,
        pixel.a as f32 / 255.0,
    )
}

#[inline]
fn from_normalized(c: f32) -> u8 {
    (f32_bound(0.0, c, 1.0) * 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use usvg::PositiveF32;

    fn make_test_pixels(count: usize) -> Vec<RGBA8> {
        let mut pixels = Vec::with_capacity(count);
        for i in 0..count {
            pixels.push(RGBA8 {
                r: (i * 17 % 256) as u8,
                g: (i * 31 % 256) as u8,
                b: (i * 59 % 256) as u8,
                a: (i * 97 % 256) as u8,
            });
        }
        pixels
    }

    fn run_bit_exact_test(matrix: &ColorMatrix, pixel_count: usize) {
        let mut pixels_naive = make_test_pixels(pixel_count);
        let mut pixels_opt = pixels_naive.clone();

        let w = pixel_count as u32;
        apply_naive(matrix, ImageRefMut::new(w, 1, &mut pixels_naive));
        apply(matrix, ImageRefMut::new(w, 1, &mut pixels_opt));

        for (i, (a, b)) in pixels_naive.iter().zip(pixels_opt.iter()).enumerate() {
            assert_eq!(*a, *b, "Mismatch at pixel {}: naive={:?} opt={:?}", i, a, b);
        }
    }

    #[test]
    fn bit_exact_full_matrix() {
        // A non-trivial 4x5 matrix
        let m = vec![
            0.5, 0.1, 0.2, 0.0, 0.05, // row 0 (R)
            0.0, 0.8, 0.1, 0.0, 0.0, // row 1 (G)
            0.1, 0.0, 0.7, 0.1, 0.02, // row 2 (B)
            0.0, 0.0, 0.0, 0.9, 0.1, // row 3 (A)
        ];
        let matrix = ColorMatrix::Matrix(m);
        run_bit_exact_test(&matrix, 1024);
    }

    #[test]
    fn bit_exact_saturate() {
        let matrix = ColorMatrix::Saturate(PositiveF32::new(0.5).unwrap());
        run_bit_exact_test(&matrix, 1024);
    }

    #[test]
    fn bit_exact_hue_rotate() {
        let matrix = ColorMatrix::HueRotate(45.0);
        run_bit_exact_test(&matrix, 1024);
    }

    #[test]
    fn bit_exact_luminance_to_alpha() {
        let matrix = ColorMatrix::LuminanceToAlpha;
        run_bit_exact_test(&matrix, 1024);
    }

    #[test]
    fn bit_exact_identity_matrix() {
        let m = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        let matrix = ColorMatrix::Matrix(m);
        run_bit_exact_test(&matrix, 1024);
    }

    #[test]
    fn bit_exact_saturate_zero() {
        let matrix = ColorMatrix::Saturate(PositiveF32::new(f32::MIN_POSITIVE).unwrap());
        run_bit_exact_test(&matrix, 1024);
    }

    #[test]
    fn bit_exact_saturate_two() {
        let matrix = ColorMatrix::Saturate(PositiveF32::new(2.0).unwrap());
        run_bit_exact_test(&matrix, 1024);
    }

    #[test]
    fn bit_exact_hue_rotate_boundary_angles() {
        for angle in &[0.0_f32, 90.0, 180.0, 270.0, 360.0, -45.0, 123.456] {
            let matrix = ColorMatrix::HueRotate(*angle);
            run_bit_exact_test(&matrix, 1024);
        }
    }

    #[test]
    fn bit_exact_extreme_matrix() {
        // Matrix with large and negative values to test clamping
        let m = vec![
            2.0, -1.0, 0.5, 0.0, -0.5, 0.0, 3.0, -0.5, 0.0, 0.2, -1.0, 0.0, 2.5, 0.0, -0.3, 0.0,
            0.0, 0.0, 1.5, -0.1,
        ];
        let matrix = ColorMatrix::Matrix(m);
        run_bit_exact_test(&matrix, 1024);
    }

    #[test]
    fn bit_exact_all_pixel_values() {
        // Test with all possible u8 values for each channel
        let mut pixels_naive: Vec<RGBA8> = Vec::with_capacity(256);
        for i in 0..=255u8 {
            pixels_naive.push(RGBA8 {
                r: i,
                g: i.wrapping_mul(3),
                b: i.wrapping_mul(7),
                a: i.wrapping_mul(13),
            });
        }
        let mut pixels_opt = pixels_naive.clone();

        let m = vec![
            0.393, 0.769, 0.189, 0.0, 0.0, // sepia-like R
            0.349, 0.686, 0.168, 0.0, 0.0, // sepia-like G
            0.272, 0.534, 0.131, 0.0, 0.0, // sepia-like B
            0.0, 0.0, 0.0, 1.0, 0.0, // A unchanged
        ];
        let matrix = ColorMatrix::Matrix(m);

        apply_naive(&matrix, ImageRefMut::new(256, 1, &mut pixels_naive));
        apply(&matrix, ImageRefMut::new(256, 1, &mut pixels_opt));

        for (i, (a, b)) in pixels_naive.iter().zip(pixels_opt.iter()).enumerate() {
            assert_eq!(*a, *b, "Mismatch at pixel {}: naive={:?} opt={:?}", i, a, b);
        }
    }
}
