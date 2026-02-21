// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{ImageRefMut, f32_bound};
use rgb::RGBA8;
use usvg::filter::{ConvolveMatrix, EdgeMode};

/// Applies a convolve matrix.
///
/// Input image pixels should have a **premultiplied alpha** when `preserve_alpha=false`.
///
/// # Allocations
///
/// This method will allocate a copy of the `src` image as a back buffer.
pub fn apply(matrix: &ConvolveMatrix, src: ImageRefMut) {
    let (rows, cols) = (matrix.matrix().rows(), matrix.matrix().columns());
    if rows <= 1 && cols <= 1 {
        // 1x1 kernel: the interior/edge split and pre-flipped kernel allocation
        // is pure overhead; fall back to the simple path.
        apply_naive(matrix, src);
    } else if rows as u32 > src.height || cols as u32 > src.width {
        // Kernel larger than image: interior region would be empty, so every
        // pixel takes the edge path and the per-pixel in_interior branch is
        // wasted overhead.
        apply_naive(matrix, src);
    } else {
        apply_general(matrix, src);
    }
}

/// The original naive implementation, used as fallback for trivial cases (1x1 kernels,
/// kernels larger than the image) where the optimized path has unnecessary overhead.
fn apply_naive(matrix: &ConvolveMatrix, src: ImageRefMut) {
    fn bound(min: i32, val: i32, max: i32) -> i32 {
        core::cmp::max(min, core::cmp::min(max, val))
    }

    let width_max = src.width as i32 - 1;
    let height_max = src.height as i32 - 1;

    let mut buf = vec![RGBA8::default(); src.data.len()];
    let mut buf = ImageRefMut::new(src.width, src.height, &mut buf);
    let mut x = 0;
    let mut y = 0;
    for in_p in src.data.iter() {
        let mut new_r = 0.0;
        let mut new_g = 0.0;
        let mut new_b = 0.0;
        let mut new_a = 0.0;
        for oy in 0..matrix.matrix().rows() {
            for ox in 0..matrix.matrix().columns() {
                let mut tx = x as i32 - matrix.matrix().target_x() as i32 + ox as i32;
                let mut ty = y as i32 - matrix.matrix().target_y() as i32 + oy as i32;

                match matrix.edge_mode() {
                    EdgeMode::None => {
                        if tx < 0 || tx > width_max || ty < 0 || ty > height_max {
                            continue;
                        }
                    }
                    EdgeMode::Duplicate => {
                        tx = bound(0, tx, width_max);
                        ty = bound(0, ty, height_max);
                    }
                    EdgeMode::Wrap => {
                        while tx < 0 {
                            tx += src.width as i32;
                        }
                        tx %= src.width as i32;

                        while ty < 0 {
                            ty += src.height as i32;
                        }
                        ty %= src.height as i32;
                    }
                }

                let k = matrix.matrix().get(
                    matrix.matrix().columns() - ox - 1,
                    matrix.matrix().rows() - oy - 1,
                );

                let p = src.pixel_at(tx as u32, ty as u32);
                new_r += (p.r as f32) / 255.0 * k;
                new_g += (p.g as f32) / 255.0 * k;
                new_b += (p.b as f32) / 255.0 * k;

                if !matrix.preserve_alpha() {
                    new_a += (p.a as f32) / 255.0 * k;
                }
            }
        }

        if matrix.preserve_alpha() {
            new_a = in_p.a as f32 / 255.0;
        } else {
            new_a = new_a / matrix.divisor().get() + matrix.bias();
        }

        let bounded_new_a = f32_bound(0.0, new_a, 1.0);

        let calc = |x| {
            let x = x / matrix.divisor().get() + matrix.bias() * new_a;

            let x = if matrix.preserve_alpha() {
                f32_bound(0.0, x, 1.0) * bounded_new_a
            } else {
                f32_bound(0.0, x, bounded_new_a)
            };

            (x * 255.0 + 0.5) as u8
        };

        let out_p = buf.pixel_at_mut(x, y);
        out_p.r = calc(new_r);
        out_p.g = calc(new_g);
        out_p.b = calc(new_b);
        out_p.a = (bounded_new_a * 255.0 + 0.5) as u8;

        x += 1;
        if x == src.width {
            x = 0;
            y += 1;
        }
    }

    // Do not use `mem::swap` because `data` referenced via FFI.
    src.data.copy_from_slice(buf.data);
}

/// Optimized general convolution that splits pixels into interior and edge regions.
/// Interior pixels skip all boundary checks (the majority for typical kernel sizes),
/// and the pre-flipped kernel avoids redundant reverse-indexing per pixel.
/// Note: true SIMD auto-vectorization is still limited by the AoS pixel layout
/// (interleaved R,G,B,A) and dynamic kernel sizes, but hoisting the preserve_alpha
/// branch out of the inner loop removes one obstacle.
fn apply_general(matrix: &ConvolveMatrix, src: ImageRefMut) {
    let width = src.width;
    let height = src.height;
    let width_i = width as i32;
    let height_i = height as i32;
    let cols = matrix.matrix().columns();
    let rows = matrix.matrix().rows();
    let target_x = matrix.matrix().target_x() as i32;
    let target_y = matrix.matrix().target_y() as i32;
    let preserve_alpha = matrix.preserve_alpha();
    let divisor = matrix.divisor().get();
    let bias = matrix.bias();

    // Pre-compute flipped kernel weights for cache-friendly sequential access.
    let mut kernel = Vec::with_capacity((rows * cols) as usize);
    for oy in 0..rows {
        for ox in 0..cols {
            kernel.push(matrix.matrix().get(cols - ox - 1, rows - oy - 1));
        }
    }

    let mut buf = vec![RGBA8::default(); src.data.len()];

    // Compute the interior region where no boundary checks are needed.
    let interior_x_start = target_x.max(0) as u32;
    let interior_x_end = (width_i - (cols as i32 - 1 - target_x)).max(0).min(width_i) as u32;
    let interior_y_start = target_y.max(0) as u32;
    let interior_y_end = (height_i - (rows as i32 - 1 - target_y))
        .max(0)
        .min(height_i) as u32;

    // Process all pixels, with fast path for interior.
    for y in 0..height {
        for x in 0..width {
            let in_p = src.data[(width * y + x) as usize];

            // Use [f32; 4] array for RGBA accumulation.
            let mut accum = [0.0f32; 4];

            let in_interior = x >= interior_x_start
                && x < interior_x_end
                && y >= interior_y_start
                && y < interior_y_end;

            if in_interior {
                // Fast path: no boundary checks needed.
                // The preserve_alpha branch is hoisted outside the inner loop
                // so that each loop body has a fixed number of accumulations,
                // giving the compiler a better chance to vectorize.
                let base_x = x as i32 - target_x;
                let base_y = y as i32 - target_y;
                if preserve_alpha {
                    // Accumulate R, G, B only (3 channels).
                    let mut ki = 0;
                    for oy in 0..rows {
                        let row_offset = ((base_y + oy as i32) as u32 * width) as usize;
                        for ox in 0..cols {
                            let px_idx = row_offset + (base_x + ox as i32) as usize;
                            let p = src.data[px_idx];
                            let k = kernel[ki];
                            ki += 1;
                            // Use / 255.0 (not * inv_255) for bit-exact match with naive.
                            accum[0] += (p.r as f32) / 255.0 * k;
                            accum[1] += (p.g as f32) / 255.0 * k;
                            accum[2] += (p.b as f32) / 255.0 * k;
                        }
                    }
                } else {
                    // Accumulate all 4 channels (R, G, B, A).
                    let mut ki = 0;
                    for oy in 0..rows {
                        let row_offset = ((base_y + oy as i32) as u32 * width) as usize;
                        for ox in 0..cols {
                            let px_idx = row_offset + (base_x + ox as i32) as usize;
                            let p = src.data[px_idx];
                            let k = kernel[ki];
                            ki += 1;
                            // Use / 255.0 (not * inv_255) for bit-exact match with naive.
                            accum[0] += (p.r as f32) / 255.0 * k;
                            accum[1] += (p.g as f32) / 255.0 * k;
                            accum[2] += (p.b as f32) / 255.0 * k;
                            accum[3] += (p.a as f32) / 255.0 * k;
                        }
                    }
                }
            } else {
                // Edge path: handle boundary conditions.
                // Same preserve_alpha split to keep the branch out of the inner loop.
                if preserve_alpha {
                    let mut ki = 0;
                    for oy in 0..rows {
                        for ox in 0..cols {
                            let mut tx = x as i32 - target_x + ox as i32;
                            let mut ty = y as i32 - target_y + oy as i32;

                            let k = kernel[ki];
                            ki += 1;

                            match matrix.edge_mode() {
                                EdgeMode::None => {
                                    if tx < 0 || tx >= width_i || ty < 0 || ty >= height_i {
                                        continue;
                                    }
                                }
                                EdgeMode::Duplicate => {
                                    tx = tx.max(0).min(width_i - 1);
                                    ty = ty.max(0).min(height_i - 1);
                                }
                                EdgeMode::Wrap => {
                                    tx = tx.rem_euclid(width_i);
                                    ty = ty.rem_euclid(height_i);
                                }
                            }

                            let p = src.data[(ty as u32 * width + tx as u32) as usize];
                            accum[0] += (p.r as f32) / 255.0 * k;
                            accum[1] += (p.g as f32) / 255.0 * k;
                            accum[2] += (p.b as f32) / 255.0 * k;
                        }
                    }
                } else {
                    let mut ki = 0;
                    for oy in 0..rows {
                        for ox in 0..cols {
                            let mut tx = x as i32 - target_x + ox as i32;
                            let mut ty = y as i32 - target_y + oy as i32;

                            let k = kernel[ki];
                            ki += 1;

                            match matrix.edge_mode() {
                                EdgeMode::None => {
                                    if tx < 0 || tx >= width_i || ty < 0 || ty >= height_i {
                                        continue;
                                    }
                                }
                                EdgeMode::Duplicate => {
                                    tx = tx.max(0).min(width_i - 1);
                                    ty = ty.max(0).min(height_i - 1);
                                }
                                EdgeMode::Wrap => {
                                    tx = tx.rem_euclid(width_i);
                                    ty = ty.rem_euclid(height_i);
                                }
                            }

                            let p = src.data[(ty as u32 * width + tx as u32) as usize];
                            accum[0] += (p.r as f32) / 255.0 * k;
                            accum[1] += (p.g as f32) / 255.0 * k;
                            accum[2] += (p.b as f32) / 255.0 * k;
                            accum[3] += (p.a as f32) / 255.0 * k;
                        }
                    }
                }
            }

            let new_a = if preserve_alpha {
                in_p.a as f32 / 255.0
            } else {
                accum[3] / divisor + bias
            };

            let bounded_new_a = f32_bound(0.0, new_a, 1.0);

            let out_idx = (width * y + x) as usize;
            let out_p = &mut buf[out_idx];

            // Compute final RGB values.
            let calc = |v: f32| -> u8 {
                let v = v / divisor + bias * new_a;
                let v = if preserve_alpha {
                    f32_bound(0.0, v, 1.0) * bounded_new_a
                } else {
                    f32_bound(0.0, v, bounded_new_a)
                };
                (v * 255.0 + 0.5) as u8
            };

            out_p.r = calc(accum[0]);
            out_p.g = calc(accum[1]);
            out_p.b = calc(accum[2]);
            out_p.a = (bounded_new_a * 255.0 + 0.5) as u8;
        }
    }

    // Do not use `mem::swap` because `data` referenced via FFI.
    src.data.copy_from_slice(&buf);
}

/// Attempt to decompose a 2D kernel into two 1D vectors (separable kernel).
///
/// A kernel K is separable if K = col_vec * row_vec^T (outer product).
/// We check this by taking the first non-zero row as the row vector,
/// then verifying all other rows are scalar multiples of it.
#[allow(dead_code)]
fn try_separate_kernel(matrix: &ConvolveMatrix) -> Option<(Vec<f32>, Vec<f32>)> {
    let cols = matrix.matrix().columns() as usize;
    let rows = matrix.matrix().rows() as usize;
    let data = matrix.matrix().data();

    // Find the first row with a non-zero element.
    let mut ref_row_idx = None;
    for r in 0..rows {
        for c in 0..cols {
            let val = data[r * cols + c];
            if val.abs() > 1e-10 {
                ref_row_idx = Some(r);
                break;
            }
        }
        if ref_row_idx.is_some() {
            break;
        }
    }

    let ref_row_idx = ref_row_idx?; // All zeros -> not useful to separate

    let ref_row: Vec<f32> = data[ref_row_idx * cols..(ref_row_idx + 1) * cols].to_vec();

    // Find a non-zero element in the reference row for normalization.
    let ref_col_idx = ref_row.iter().position(|&v| v.abs() > 1e-10)?;
    let ref_val = ref_row[ref_col_idx];

    // Build column vector: each row's scale relative to the reference row.
    let mut col_vec = vec![0.0f32; rows];
    for r in 0..rows {
        let row_val = data[r * cols + ref_col_idx];
        col_vec[r] = row_val / ref_val;
    }

    // Normalize: row_vec = ref_row (the actual row values).
    let row_vec = ref_row;

    // Verify separability: for every element, K[r][c] should equal col_vec[r] * row_vec[c].
    let tolerance = 1e-6;
    for r in 0..rows {
        for c in 0..cols {
            let expected = col_vec[r] * row_vec[c];
            let actual = data[r * cols + c];
            if (expected - actual).abs() > tolerance * actual.abs().max(1.0) {
                return None;
            }
        }
    }

    Some((col_vec, row_vec))
}

/// Apply convolution using separable kernel decomposition (two 1D passes).
///
/// This is O(N * (K_cols + K_rows)) instead of O(N * K_cols * K_rows).
///
/// To maintain bit-exact results with the general path, we must replicate
/// the exact same floating-point arithmetic. The 2D convolution computes:
///   sum += pixel[r][c] * col_vec[r] * row_vec[c]
///
/// By factoring: sum = sum_r(col_vec[r] * sum_c(pixel[r][c] * row_vec[c]))
///
/// However, floating-point addition is not associative, so reordering the
/// summation changes results. For bit-exact match, we'd need the horizontal
/// pass first to produce intermediate f32 results, then the vertical pass.
///
/// Since bit-exact match with the naive 2D loop is required, we use a
/// carefully ordered two-pass approach that replicates the accumulation order:
///   Pass 1 (horizontal): for each (x,y), compute sum_c(pixel[y][c+x-tx] * row_vec[c])
///   Pass 2 (vertical): for each (x,y), compute sum_r(col_vec[r] * horiz[y+r-ty][x])
///
/// The 2D naive loop does: for oy { for ox { accum += pixel * (col[oy]*row[ox]) } }
/// which equals: for oy { col[oy] * for ox { accum_inner += pixel * row[ox] } }
/// The horizontal-then-vertical decomposition matches this summation order
/// if the vertical pass multiplies by col_vec[r] and the horizontal pass
/// produces the inner sum.
///
/// IMPORTANT: Due to floating-point non-associativity, the separable path may
/// produce very slightly different results from the naive 2D path. However,
/// both are valid interpretations of the convolution. We verify correctness
/// independently.
#[allow(dead_code)]
fn apply_separable(matrix: &ConvolveMatrix, col_vec: &[f32], row_vec: &[f32], src: ImageRefMut) {
    let width = src.width;
    let height = src.height;
    let width_i = width as i32;
    let height_i = height as i32;
    let cols = matrix.matrix().columns() as i32;
    let rows = matrix.matrix().rows() as i32;
    let target_x = matrix.matrix().target_x() as i32;
    let target_y = matrix.matrix().target_y() as i32;
    let preserve_alpha = matrix.preserve_alpha();
    let divisor = matrix.divisor().get();
    let bias = matrix.bias();

    // Flip the kernel vectors to match the flipped indexing in the original.
    // Original: kernel.get(cols-ox-1, rows-oy-1)
    // For separable: row_vec is indexed as row_vec[cols-ox-1], col_vec as col_vec[rows-oy-1]
    let flipped_row: Vec<f32> = row_vec.iter().rev().copied().collect();
    let flipped_col: Vec<f32> = col_vec.iter().rev().copied().collect();

    // Pass 1: Horizontal convolution with row_vec.
    // For each pixel (x,y), compute the weighted sum along the row direction.
    // Result is stored as [f32; 4] per pixel (R, G, B, A).
    let total_pixels = (width * height) as usize;
    let mut horiz = vec![[0.0f32; 4]; total_pixels];

    // Interior bounds for horizontal pass.
    let h_interior_x_start = target_x.max(0) as u32;
    let h_interior_x_end = (width_i - (cols - 1 - target_x)).max(0).min(width_i) as u32;

    for y in 0..height {
        let row_base = (y * width) as usize;
        for x in 0..width {
            let mut accum = [0.0f32; 4];

            let in_interior = x >= h_interior_x_start && x < h_interior_x_end;

            if in_interior {
                let base_x = (x as i32 - target_x) as usize;
                for ox in 0..cols as usize {
                    let px_idx = row_base + base_x + ox;
                    let p = src.data[px_idx];
                    let k = flipped_row[ox];
                    accum[0] += (p.r as f32) / 255.0 * k;
                    accum[1] += (p.g as f32) / 255.0 * k;
                    accum[2] += (p.b as f32) / 255.0 * k;
                    if !preserve_alpha {
                        accum[3] += (p.a as f32) / 255.0 * k;
                    }
                }
            } else {
                for ox in 0..cols {
                    let mut tx = x as i32 - target_x + ox;
                    match matrix.edge_mode() {
                        EdgeMode::None => {
                            if tx < 0 || tx >= width_i {
                                continue;
                            }
                        }
                        EdgeMode::Duplicate => {
                            tx = tx.max(0).min(width_i - 1);
                        }
                        EdgeMode::Wrap => {
                            tx = tx.rem_euclid(width_i);
                        }
                    }
                    let px_idx = row_base + tx as usize;
                    let p = src.data[px_idx];
                    let k = flipped_row[ox as usize];
                    accum[0] += (p.r as f32) / 255.0 * k;
                    accum[1] += (p.g as f32) / 255.0 * k;
                    accum[2] += (p.b as f32) / 255.0 * k;
                    if !preserve_alpha {
                        accum[3] += (p.a as f32) / 255.0 * k;
                    }
                }
            }

            horiz[row_base + x as usize] = accum;
        }
    }

    // Pass 2: Vertical convolution with col_vec + final output computation.
    let mut buf = vec![RGBA8::default(); src.data.len()];

    // Interior bounds for vertical pass.
    let v_interior_y_start = target_y.max(0) as u32;
    let v_interior_y_end = (height_i - (rows - 1 - target_y)).max(0).min(height_i) as u32;

    for y in 0..height {
        for x in 0..width {
            let in_p = src.data[(width * y + x) as usize];
            let mut accum = [0.0f32; 4];

            let in_interior = y >= v_interior_y_start && y < v_interior_y_end;

            if in_interior {
                let base_y = (y as i32 - target_y) as u32;
                for oy in 0..rows as u32 {
                    let h_idx = ((base_y + oy) * width + x) as usize;
                    let h = horiz[h_idx];
                    let k = flipped_col[oy as usize];
                    accum[0] += h[0] * k;
                    accum[1] += h[1] * k;
                    accum[2] += h[2] * k;
                    if !preserve_alpha {
                        accum[3] += h[3] * k;
                    }
                }
            } else {
                for oy in 0..rows {
                    let mut ty = y as i32 - target_y + oy;
                    match matrix.edge_mode() {
                        EdgeMode::None => {
                            if ty < 0 || ty >= height_i {
                                continue;
                            }
                        }
                        EdgeMode::Duplicate => {
                            ty = ty.max(0).min(height_i - 1);
                        }
                        EdgeMode::Wrap => {
                            ty = ty.rem_euclid(height_i);
                        }
                    }
                    let h_idx = (ty as u32 * width + x) as usize;
                    let h = horiz[h_idx];
                    let k = flipped_col[oy as usize];
                    accum[0] += h[0] * k;
                    accum[1] += h[1] * k;
                    accum[2] += h[2] * k;
                    if !preserve_alpha {
                        accum[3] += h[3] * k;
                    }
                }
            }

            let new_a = if preserve_alpha {
                in_p.a as f32 / 255.0
            } else {
                accum[3] / divisor + bias
            };

            let bounded_new_a = f32_bound(0.0, new_a, 1.0);

            let out_idx = (width * y + x) as usize;
            let out_p = &mut buf[out_idx];

            let calc = |v: f32| -> u8 {
                let v = v / divisor + bias * new_a;
                let v = if preserve_alpha {
                    f32_bound(0.0, v, 1.0) * bounded_new_a
                } else {
                    f32_bound(0.0, v, bounded_new_a)
                };
                (v * 255.0 + 0.5) as u8
            };

            out_p.r = calc(accum[0]);
            out_p.g = calc(accum[1]);
            out_p.b = calc(accum[2]);
            out_p.a = (bounded_new_a * 255.0 + 0.5) as u8;
        }
    }

    // Do not use `mem::swap` because `data` referenced via FFI.
    src.data.copy_from_slice(&buf);
}

#[cfg(test)]
mod tests {
    use super::*;
    use usvg::filter::ConvolveMatrixData;

    /// Helper to create a ConvolveMatrix for testing.
    fn make_convolve_matrix(
        cols: u32,
        rows: u32,
        target_x: u32,
        target_y: u32,
        data: Vec<f32>,
        divisor: f32,
        bias: f32,
        edge_mode: EdgeMode,
        preserve_alpha: bool,
    ) -> ConvolveMatrix {
        ConvolveMatrix::new(
            usvg::filter::Input::SourceGraphic,
            ConvolveMatrixData::new(target_x, target_y, cols, rows, data).unwrap(),
            usvg::NonZeroF32::new(divisor).unwrap(),
            bias,
            edge_mode,
            preserve_alpha,
        )
    }

    /// Generate a test image with deterministic pixel values.
    fn make_test_image(width: u32, height: u32, seed: u8) -> Vec<RGBA8> {
        let mut data = Vec::with_capacity((width * height) as usize);
        for i in 0..(width * height) {
            let v = ((i as u8).wrapping_mul(17).wrapping_add(seed)) as u8;
            data.push(RGBA8 {
                r: v,
                g: v.wrapping_add(50),
                b: v.wrapping_add(100),
                a: v.wrapping_add(150),
            });
        }
        data
    }

    /// Run both naive and general implementations and verify bit-exact match.
    fn verify_general_matches_naive(matrix: &ConvolveMatrix, width: u32, height: u32, seed: u8) {
        let data_naive = make_test_image(width, height, seed);
        let data_opt = data_naive.clone();

        let mut buf_naive = data_naive;
        let mut buf_opt = data_opt;

        apply_naive(matrix, ImageRefMut::new(width, height, &mut buf_naive));
        apply_general(matrix, ImageRefMut::new(width, height, &mut buf_opt));

        for (i, (n, o)) in buf_naive.iter().zip(buf_opt.iter()).enumerate() {
            assert_eq!(
                n, o,
                "Mismatch at pixel {}: naive={:?} vs optimized={:?}",
                i, n, o
            );
        }
    }

    #[test]
    fn test_general_matches_naive_3x3_duplicate() {
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
            1.0,
            0.0,
            EdgeMode::Duplicate,
            false,
        );
        verify_general_matches_naive(&m, 16, 16, 42);
    }

    #[test]
    fn test_general_matches_naive_3x3_none() {
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            9.0,
            0.0,
            EdgeMode::None,
            false,
        );
        verify_general_matches_naive(&m, 16, 16, 7);
    }

    #[test]
    fn test_general_matches_naive_3x3_wrap() {
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
            16.0,
            0.0,
            EdgeMode::Wrap,
            false,
        );
        verify_general_matches_naive(&m, 16, 16, 99);
    }

    #[test]
    fn test_general_matches_naive_5x5() {
        let m = make_convolve_matrix(
            5,
            5,
            2,
            2,
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            25.0,
            0.0,
            EdgeMode::Duplicate,
            false,
        );
        verify_general_matches_naive(&m, 32, 32, 13);
    }

    #[test]
    fn test_general_matches_naive_preserve_alpha() {
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
            1.0,
            0.0,
            EdgeMode::Duplicate,
            true,
        );
        verify_general_matches_naive(&m, 16, 16, 55);
    }

    #[test]
    fn test_general_matches_naive_with_bias() {
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            9.0,
            0.1,
            EdgeMode::Duplicate,
            false,
        );
        verify_general_matches_naive(&m, 16, 16, 77);
    }

    #[test]
    fn test_general_matches_naive_asymmetric_kernel() {
        let m = make_convolve_matrix(
            3,
            5,
            1,
            2,
            vec![
                1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 3.0, 0.0, -3.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0,
            ],
            1.0,
            0.5,
            EdgeMode::Duplicate,
            false,
        );
        verify_general_matches_naive(&m, 20, 20, 33);
    }

    #[test]
    fn test_general_matches_naive_off_center_target() {
        let m = make_convolve_matrix(
            3,
            3,
            0,
            0,
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            9.0,
            0.0,
            EdgeMode::Duplicate,
            false,
        );
        verify_general_matches_naive(&m, 16, 16, 11);
    }

    #[test]
    fn test_general_matches_naive_max_target() {
        let m = make_convolve_matrix(
            3,
            3,
            2,
            2,
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            9.0,
            0.0,
            EdgeMode::Duplicate,
            false,
        );
        verify_general_matches_naive(&m, 16, 16, 22);
    }

    #[test]
    fn test_general_matches_naive_1x1() {
        let m = make_convolve_matrix(1, 1, 0, 0, vec![2.0], 2.0, 0.0, EdgeMode::Duplicate, false);
        verify_general_matches_naive(&m, 8, 8, 44);
    }

    #[test]
    fn test_general_matches_naive_large_kernel_7x7() {
        let m = make_convolve_matrix(
            7,
            7,
            3,
            3,
            vec![1.0; 49],
            49.0,
            0.0,
            EdgeMode::Duplicate,
            false,
        );
        verify_general_matches_naive(&m, 32, 32, 88);
    }

    #[test]
    fn test_general_matches_naive_9x9() {
        let m = make_convolve_matrix(9, 9, 4, 4, vec![1.0; 81], 81.0, 0.0, EdgeMode::Wrap, false);
        verify_general_matches_naive(&m, 32, 32, 66);
    }

    #[test]
    fn test_separable_detection_uniform() {
        // Uniform kernel is separable: [1,1,1]^T * [1,1,1]
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            vec![1.0; 9],
            9.0,
            0.0,
            EdgeMode::Duplicate,
            false,
        );
        let result = try_separate_kernel(&m);
        assert!(result.is_some(), "Uniform 3x3 kernel should be separable");
    }

    #[test]
    fn test_separable_detection_gaussian() {
        // Gaussian-like kernel [1,2,1]^T * [1,2,1]
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
            16.0,
            0.0,
            EdgeMode::Duplicate,
            false,
        );
        let result = try_separate_kernel(&m);
        assert!(result.is_some(), "Gaussian 3x3 kernel should be separable");
    }

    #[test]
    fn test_separable_detection_non_separable() {
        // Laplacian kernel is NOT separable
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            vec![0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
            1.0,
            0.0,
            EdgeMode::Duplicate,
            false,
        );
        let result = try_separate_kernel(&m);
        assert!(result.is_none(), "Laplacian kernel should NOT be separable");
    }

    /// Helper: run separable path on the given matrix and image.
    fn run_separable(m: &ConvolveMatrix, width: u32, height: u32, data: &mut [RGBA8]) {
        let (col_vec, row_vec) = try_separate_kernel(m).expect("Kernel must be separable");
        apply_separable(m, &col_vec, &row_vec, ImageRefMut::new(width, height, data));
    }

    #[test]
    fn test_separable_vs_naive_gaussian() {
        // Gaussian [1,2,1]^T * [1,2,1] - separable
        let kernel_data = vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            kernel_data,
            16.0,
            0.0,
            EdgeMode::Duplicate,
            false,
        );

        let data1 = make_test_image(32, 32, 42);
        let data2 = data1.clone();

        let mut buf_naive = data1;
        let mut buf_sep = data2;

        apply_naive(&m, ImageRefMut::new(32, 32, &mut buf_naive));
        run_separable(&m, 32, 32, &mut buf_sep);

        // Separable path may differ by tiny floating-point amounts.
        // Allow up to 1 unit of difference per channel.
        for (i, (n, s)) in buf_naive.iter().zip(buf_sep.iter()).enumerate() {
            let dr = (n.r as i16 - s.r as i16).unsigned_abs();
            let dg = (n.g as i16 - s.g as i16).unsigned_abs();
            let db = (n.b as i16 - s.b as i16).unsigned_abs();
            let da = (n.a as i16 - s.a as i16).unsigned_abs();
            assert!(
                dr <= 1 && dg <= 1 && db <= 1 && da <= 1,
                "Mismatch at pixel {}: naive={:?} vs separable={:?} (diff: r={}, g={}, b={}, a={})",
                i,
                n,
                s,
                dr,
                dg,
                db,
                da
            );
        }
    }

    #[test]
    fn test_separable_vs_naive_5x5_uniform() {
        let m = make_convolve_matrix(
            5,
            5,
            2,
            2,
            vec![1.0; 25],
            25.0,
            0.0,
            EdgeMode::Duplicate,
            false,
        );

        let data1 = make_test_image(32, 32, 77);
        let data2 = data1.clone();
        let mut buf_naive = data1;
        let mut buf_sep = data2;

        apply_naive(&m, ImageRefMut::new(32, 32, &mut buf_naive));
        run_separable(&m, 32, 32, &mut buf_sep);

        for (i, (n, s)) in buf_naive.iter().zip(buf_sep.iter()).enumerate() {
            let dr = (n.r as i16 - s.r as i16).unsigned_abs();
            let dg = (n.g as i16 - s.g as i16).unsigned_abs();
            let db = (n.b as i16 - s.b as i16).unsigned_abs();
            let da = (n.a as i16 - s.a as i16).unsigned_abs();
            assert!(
                dr <= 1 && dg <= 1 && db <= 1 && da <= 1,
                "Mismatch at pixel {}: naive={:?} vs separable={:?} (diff: r={}, g={}, b={}, a={})",
                i,
                n,
                s,
                dr,
                dg,
                db,
                da
            );
        }
    }

    #[test]
    fn test_separable_preserve_alpha() {
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
            16.0,
            0.0,
            EdgeMode::Duplicate,
            true,
        );

        let data1 = make_test_image(16, 16, 55);
        let data2 = data1.clone();
        let mut buf_naive = data1;
        let mut buf_sep = data2;

        apply_naive(&m, ImageRefMut::new(16, 16, &mut buf_naive));
        run_separable(&m, 16, 16, &mut buf_sep);

        for (i, (n, s)) in buf_naive.iter().zip(buf_sep.iter()).enumerate() {
            let dr = (n.r as i16 - s.r as i16).unsigned_abs();
            let dg = (n.g as i16 - s.g as i16).unsigned_abs();
            let db = (n.b as i16 - s.b as i16).unsigned_abs();
            let da = (n.a as i16 - s.a as i16).unsigned_abs();
            assert!(
                dr <= 1 && dg <= 1 && db <= 1 && da <= 1,
                "Mismatch at pixel {}: naive={:?} vs separable={:?}",
                i,
                n,
                s
            );
        }
    }

    #[test]
    fn test_separable_with_bias() {
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
            16.0,
            0.1,
            EdgeMode::Duplicate,
            false,
        );

        let data1 = make_test_image(16, 16, 33);
        let data2 = data1.clone();
        let mut buf_naive = data1;
        let mut buf_sep = data2;

        apply_naive(&m, ImageRefMut::new(16, 16, &mut buf_naive));
        run_separable(&m, 16, 16, &mut buf_sep);

        for (i, (n, s)) in buf_naive.iter().zip(buf_sep.iter()).enumerate() {
            let dr = (n.r as i16 - s.r as i16).unsigned_abs();
            let dg = (n.g as i16 - s.g as i16).unsigned_abs();
            let db = (n.b as i16 - s.b as i16).unsigned_abs();
            let da = (n.a as i16 - s.a as i16).unsigned_abs();
            assert!(
                dr <= 1 && dg <= 1 && db <= 1 && da <= 1,
                "Mismatch at pixel {}: naive={:?} vs separable={:?}",
                i,
                n,
                s
            );
        }
    }

    #[test]
    fn test_separable_edge_mode_none() {
        let m = make_convolve_matrix(3, 3, 1, 1, vec![1.0; 9], 9.0, 0.0, EdgeMode::None, false);

        let data1 = make_test_image(16, 16, 11);
        let data2 = data1.clone();
        let mut buf_naive = data1;
        let mut buf_sep = data2;

        apply_naive(&m, ImageRefMut::new(16, 16, &mut buf_naive));
        run_separable(&m, 16, 16, &mut buf_sep);

        for (i, (n, s)) in buf_naive.iter().zip(buf_sep.iter()).enumerate() {
            let dr = (n.r as i16 - s.r as i16).unsigned_abs();
            let dg = (n.g as i16 - s.g as i16).unsigned_abs();
            let db = (n.b as i16 - s.b as i16).unsigned_abs();
            let da = (n.a as i16 - s.a as i16).unsigned_abs();
            assert!(
                dr <= 1 && dg <= 1 && db <= 1 && da <= 1,
                "Mismatch at pixel {}: naive={:?} vs separable={:?}",
                i,
                n,
                s
            );
        }
    }

    #[test]
    fn test_separable_edge_mode_wrap() {
        let m = make_convolve_matrix(
            3,
            3,
            1,
            1,
            vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
            16.0,
            0.0,
            EdgeMode::Wrap,
            false,
        );

        let data1 = make_test_image(16, 16, 99);
        let data2 = data1.clone();
        let mut buf_naive = data1;
        let mut buf_sep = data2;

        apply_naive(&m, ImageRefMut::new(16, 16, &mut buf_naive));
        run_separable(&m, 16, 16, &mut buf_sep);

        for (i, (n, s)) in buf_naive.iter().zip(buf_sep.iter()).enumerate() {
            let dr = (n.r as i16 - s.r as i16).unsigned_abs();
            let dg = (n.g as i16 - s.g as i16).unsigned_abs();
            let db = (n.b as i16 - s.b as i16).unsigned_abs();
            let da = (n.a as i16 - s.a as i16).unsigned_abs();
            assert!(
                dr <= 1 && dg <= 1 && db <= 1 && da <= 1,
                "Mismatch at pixel {}: naive={:?} vs separable={:?}",
                i,
                n,
                s
            );
        }
    }

    /// Exhaustive test: all edge modes × preserve_alpha × various kernel sizes.
    #[test]
    fn test_general_exhaustive() {
        let edge_modes = [EdgeMode::None, EdgeMode::Duplicate, EdgeMode::Wrap];
        let preserve_alphas = [false, true];
        let kernel_configs: Vec<(u32, u32, u32, u32)> = vec![
            (1, 1, 0, 0),
            (3, 3, 1, 1),
            (5, 5, 2, 2),
            (3, 5, 1, 2),
            (5, 3, 2, 1),
            (3, 3, 0, 0),
            (3, 3, 2, 2),
        ];

        for &em in &edge_modes {
            for &pa in &preserve_alphas {
                for &(cols, rows, tx, ty) in &kernel_configs {
                    let n = (cols * rows) as usize;
                    let data: Vec<f32> = (0..n)
                        .map(|i| (i as f32 * 0.5) - (n as f32 * 0.25))
                        .collect();
                    let divisor = data.iter().sum::<f32>().abs().max(1.0);
                    let m = make_convolve_matrix(cols, rows, tx, ty, data, divisor, 0.0, em, pa);
                    verify_general_matches_naive(&m, 16, 16, 42);
                }
            }
        }
    }

    /// Helper to run `apply()` (the public dispatcher) and compare against `apply_naive()`.
    fn verify_apply_matches_naive(matrix: &ConvolveMatrix, width: u32, height: u32, seed: u8) {
        let data_naive = make_test_image(width, height, seed);
        let data_apply = data_naive.clone();

        let mut buf_naive = data_naive;
        let mut buf_apply = data_apply;

        apply_naive(matrix, ImageRefMut::new(width, height, &mut buf_naive));
        apply(matrix, ImageRefMut::new(width, height, &mut buf_apply));

        for (i, (n, a)) in buf_naive.iter().zip(buf_apply.iter()).enumerate() {
            assert_eq!(
                n, a,
                "Mismatch at pixel {}: naive={:?} vs apply={:?}",
                i, n, a
            );
        }
    }

    /// Test that apply() correctly dispatches 1x1 kernels through the naive fallback.
    #[test]
    fn test_apply_1x1_uses_naive_fallback() {
        let m = make_convolve_matrix(1, 1, 0, 0, vec![2.0], 2.0, 0.0, EdgeMode::Duplicate, false);
        verify_apply_matches_naive(&m, 8, 8, 44);
    }

    /// Test that apply() correctly dispatches kernels larger than the image
    /// through the naive fallback.
    #[test]
    fn test_apply_oversized_kernel_uses_naive_fallback() {
        // 5x5 kernel on a 4x4 image: cols (5) > width (4)
        let m = make_convolve_matrix(
            5,
            5,
            2,
            2,
            vec![1.0; 25],
            25.0,
            0.0,
            EdgeMode::Duplicate,
            false,
        );
        verify_apply_matches_naive(&m, 4, 4, 10);
    }

    /// Test that apply() correctly dispatches when only one kernel dimension
    /// exceeds the image dimension.
    #[test]
    fn test_apply_oversized_kernel_one_dimension() {
        // 3x7 kernel on a 8x5 image: rows (7) > height (5), cols (3) <= width (8)
        let m = make_convolve_matrix(3, 7, 1, 3, vec![1.0; 21], 21.0, 0.0, EdgeMode::Wrap, false);
        verify_apply_matches_naive(&m, 8, 5, 77);
    }
}
