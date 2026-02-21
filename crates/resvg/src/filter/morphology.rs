// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::ImageRefMut;
use rgb::RGBA8;
use usvg::filter::MorphologyOperator;

// For small kernels, the naive O(n*r^2) approach is faster than vHGW due to
// lower overhead. This threshold is the maximum kernel area (columns * rows)
// for which we use the naive path.
const NAIVE_KERNEL_AREA_THRESHOLD: u32 = 32;

/// Applies a morphology filter.
///
/// `src` pixels should have a **premultiplied alpha**.
///
/// # Allocations
///
/// This method will allocate a copy of the `src` image as a back buffer.
pub fn apply(operator: MorphologyOperator, rx: f32, ry: f32, src: ImageRefMut) {
    let columns = std::cmp::min(rx.ceil() as u32 * 2, src.width);
    let rows = std::cmp::min(ry.ceil() as u32 * 2, src.height);

    if columns * rows <= NAIVE_KERNEL_AREA_THRESHOLD {
        apply_naive(operator, columns, rows, src);
    } else {
        match operator {
            MorphologyOperator::Erode => apply_vhgw::<ErodeOp>(columns, rows, src),
            MorphologyOperator::Dilate => apply_vhgw::<DilateOp>(columns, rows, src),
        }
    }
}

/// Original naive O(n × r²) morphology — used as fallback for small kernels.
fn apply_naive(operator: MorphologyOperator, columns: u32, rows: u32, src: ImageRefMut) {
    let target_x = (columns as f32 / 2.0).floor() as u32;
    let target_y = (rows as f32 / 2.0).floor() as u32;

    let width_max = src.width as i32 - 1;
    let height_max = src.height as i32 - 1;

    let mut buf = vec![RGBA8::default(); src.data.len()];
    let mut buf = ImageRefMut::new(src.width, src.height, &mut buf);
    let mut x = 0;
    let mut y = 0;
    for _ in src.data.iter() {
        let mut new_p = RGBA8::default();
        if operator == MorphologyOperator::Erode {
            new_p.r = 255;
            new_p.g = 255;
            new_p.b = 255;
            new_p.a = 255;
        }

        for oy in 0..rows {
            for ox in 0..columns {
                let tx = x as i32 - target_x as i32 + ox as i32;
                let ty = y as i32 - target_y as i32 + oy as i32;

                if tx < 0 || tx > width_max || ty < 0 || ty > height_max {
                    continue;
                }

                let p = src.pixel_at(tx as u32, ty as u32);
                if operator == MorphologyOperator::Erode {
                    new_p.r = std::cmp::min(p.r, new_p.r);
                    new_p.g = std::cmp::min(p.g, new_p.g);
                    new_p.b = std::cmp::min(p.b, new_p.b);
                    new_p.a = std::cmp::min(p.a, new_p.a);
                } else {
                    new_p.r = std::cmp::max(p.r, new_p.r);
                    new_p.g = std::cmp::max(p.g, new_p.g);
                    new_p.b = std::cmp::max(p.b, new_p.b);
                    new_p.a = std::cmp::max(p.a, new_p.a);
                }
            }
        }

        *buf.pixel_at_mut(x, y) = new_p;

        x += 1;
        if x == src.width {
            x = 0;
            y += 1;
        }
    }

    // Do not use `mem::swap` because `data` referenced via FFI.
    src.data.copy_from_slice(buf.data);
}

// ---------------------------------------------------------------------------
// van Herk/Gil-Werman (vHGW) separable morphology — O(n) regardless of radius
// ---------------------------------------------------------------------------
//
// 2D rectangular min/max is separable: min_rect(img) = min_y(min_x(img)).
// Each 1D pass uses the vHGW algorithm with identity-padded boundaries:
//
// 1. Pad input with (win-1) identity elements on each side
// 2. Forward prefix scan within blocks of size `win`
// 3. Backward suffix scan within blocks of size `win`
// 4. Merge: output[i] = op(suffix[left], prefix[right])
//
// Cost: ~3 comparisons per pixel per channel, independent of window size.
//
// Using [u8; 4] for pixel data and a monomorphized trait lets LLVM
// auto-vectorize the inner loops (SSE/AVX pminub/pmaxub, NEON vmin/vmax).

trait MorphOp {
    const IDENTITY: [u8; 4];
    fn op(a: [u8; 4], b: [u8; 4]) -> [u8; 4];
}

struct ErodeOp;
impl MorphOp for ErodeOp {
    const IDENTITY: [u8; 4] = [255, 255, 255, 255];

    #[inline(always)]
    fn op(a: [u8; 4], b: [u8; 4]) -> [u8; 4] {
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].min(b[2]),
            a[3].min(b[3]),
        ]
    }
}

struct DilateOp;
impl MorphOp for DilateOp {
    const IDENTITY: [u8; 4] = [0, 0, 0, 0];

    #[inline(always)]
    fn op(a: [u8; 4], b: [u8; 4]) -> [u8; 4] {
        [
            a[0].max(b[0]),
            a[1].max(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
        ]
    }
}

/// 1D van Herk/Gil-Werman pass.
///
/// - `input`: source pixels of length `n`
/// - `output`: destination slice of length `n`
/// - `win`: window size (columns or rows)
/// - `target`: offset of the target pixel within the window (floor(win/2))
/// - `prefix`, `suffix`, `padded`: pre-allocated scratch buffers
///
/// All buffers are resized as needed by the caller.
fn vhgw_1d<Op: MorphOp>(
    input: &[[u8; 4]],
    output: &mut [[u8; 4]],
    win: usize,
    target: usize,
    prefix: &mut Vec<[u8; 4]>,
    suffix: &mut Vec<[u8; 4]>,
    padded: &mut Vec<[u8; 4]>,
) {
    let n = input.len();
    debug_assert_eq!(output.len(), n);

    if win <= 1 {
        output.copy_from_slice(input);
        return;
    }

    let pad = win - 1;
    let padded_len = n + 2 * pad;

    padded.resize(padded_len, Op::IDENTITY);
    prefix.resize(padded_len, Op::IDENTITY);
    suffix.resize(padded_len, Op::IDENTITY);

    // Fill padded: identity margins + input in the middle
    padded[..pad].fill(Op::IDENTITY);
    padded[pad..pad + n].copy_from_slice(input);
    padded[pad + n..].fill(Op::IDENTITY);

    // Forward prefix scan within blocks of size `win`
    for block_start in (0..padded_len).step_by(win) {
        let block_end = (block_start + win).min(padded_len);
        prefix[block_start] = padded[block_start];
        for i in (block_start + 1)..block_end {
            prefix[i] = Op::op(prefix[i - 1], padded[i]);
        }
    }

    // Backward suffix scan within blocks of size `win`
    for block_start in (0..padded_len).step_by(win) {
        let block_end = (block_start + win).min(padded_len);
        suffix[block_end - 1] = padded[block_end - 1];
        for i in (block_start..block_end - 1).rev() {
            suffix[i] = Op::op(suffix[i + 1], padded[i]);
        }
    }

    // Merge pass
    for i in 0..n {
        let left = pad + i - target;
        let right = left + win - 1;
        output[i] = Op::op(suffix[left], prefix[right]);
    }
}

/// Separable vHGW morphology: horizontal pass then vertical pass.
fn apply_vhgw<Op: MorphOp>(columns: u32, rows: u32, src: ImageRefMut) {
    let width = src.width as usize;
    let height = src.height as usize;
    let target_x = (columns as f32 / 2.0).floor() as usize;
    let target_y = (rows as f32 / 2.0).floor() as usize;
    let win_x = columns as usize;
    let win_y = rows as usize;

    // Work in [u8; 4] throughout to avoid repeated RGBA8 ↔ [u8; 4] conversions.
    // This also enables better auto-vectorization since LLVM sees contiguous u8 ops.
    let pixel_count = width * height;
    let mut buf = vec![[0u8; 4]; pixel_count];

    // Pre-allocate scratch buffers (reused across all rows/columns)
    let max_dim = width.max(height);
    let max_win = win_x.max(win_y);
    let max_padded = max_dim + 2 * max_win;
    let mut prefix = Vec::with_capacity(max_padded);
    let mut suffix = Vec::with_capacity(max_padded);
    let mut padded = Vec::with_capacity(max_padded);

    // Horizontal pass: src → buf (row-contiguous access — cache friendly)
    {
        let mut row_out = vec![[0u8; 4]; width];
        for y in 0..height {
            let row_start = y * width;
            // Convert src row to [u8; 4] in-place within buf, then use as input
            for x in 0..width {
                let p = src.data[row_start + x];
                buf[row_start + x] = [p.r, p.g, p.b, p.a];
            }
            vhgw_1d::<Op>(
                &buf[row_start..row_start + width],
                &mut row_out,
                win_x, target_x,
                &mut prefix, &mut suffix, &mut padded,
            );
            buf[row_start..row_start + width].copy_from_slice(&row_out);
        }
    }

    // Vertical pass: buf → src
    // Process columns in tiles for cache friendliness. Each tile extracts
    // TILE_WIDTH columns into a contiguous temporary, runs vHGW on each,
    // and writes results back.
    const TILE_WIDTH: usize = 8;
    {
        let mut col_in = vec![[0u8; 4]; height];
        let mut col_out = vec![[0u8; 4]; height];
        let mut x = 0;
        while x < width {
            let tile_end = (x + TILE_WIDTH).min(width);
            for col in x..tile_end {
                // Gather column
                for y in 0..height {
                    col_in[y] = buf[y * width + col];
                }
                vhgw_1d::<Op>(
                    &col_in, &mut col_out,
                    win_y, target_y,
                    &mut prefix, &mut suffix, &mut padded,
                );
                // Scatter column back to src
                for y in 0..height {
                    let p = col_out[y];
                    src.data[y * width + col] = RGBA8 { r: p[0], g: p[1], b: p[2], a: p[3] };
                }
            }
            x = tile_end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run naive morphology on a copy of the data
    fn run_naive(operator: MorphologyOperator, rx: f32, ry: f32, data: &mut [RGBA8], w: u32, h: u32) {
        let columns = std::cmp::min(rx.ceil() as u32 * 2, w);
        let rows = std::cmp::min(ry.ceil() as u32 * 2, h);
        let src = ImageRefMut::new(w, h, data);
        apply_naive(operator, columns, rows, src);
    }

    /// Run vHGW morphology on a copy of the data
    fn run_vhgw(operator: MorphologyOperator, rx: f32, ry: f32, data: &mut [RGBA8], w: u32, h: u32) {
        let columns = std::cmp::min(rx.ceil() as u32 * 2, w);
        let rows = std::cmp::min(ry.ceil() as u32 * 2, h);
        let src = ImageRefMut::new(w, h, data);
        match operator {
            MorphologyOperator::Erode => apply_vhgw::<ErodeOp>(columns, rows, src),
            MorphologyOperator::Dilate => apply_vhgw::<DilateOp>(columns, rows, src),
        }
    }

    /// Compare naive vs vHGW for given parameters
    fn assert_bit_exact(operator: MorphologyOperator, rx: f32, ry: f32, data: &[RGBA8], w: u32, h: u32) {
        let mut naive_data = data.to_vec();
        let mut vhgw_data = data.to_vec();
        run_naive(operator, rx, ry, &mut naive_data, w, h);
        run_vhgw(operator, rx, ry, &mut vhgw_data, w, h);
        assert_eq!(naive_data, vhgw_data,
            "Mismatch for {:?} rx={} ry={} {}x{}", operator, rx, ry, w, h);
    }

    fn make_gradient(w: u32, h: u32) -> Vec<RGBA8> {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = ((x + y) % 256) as u8;
                data.push(RGBA8 { r: v, g: v.wrapping_mul(3), b: v.wrapping_mul(7), a: 255 });
            }
        }
        data
    }

    fn make_random(w: u32, h: u32, seed: u64) -> Vec<RGBA8> {
        let mut data = Vec::with_capacity((w * h) as usize);
        let mut state = seed;
        for _ in 0..(w * h) {
            // Simple xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bytes = state.to_le_bytes();
            data.push(RGBA8 { r: bytes[0], g: bytes[1], b: bytes[2], a: bytes[3] });
        }
        data
    }

    #[test]
    fn vhgw_matches_naive_erode_various_radii() {
        let data = make_random(50, 50, 12345);
        for r in [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0] {
            assert_bit_exact(MorphologyOperator::Erode, r, r, &data, 50, 50);
        }
    }

    #[test]
    fn vhgw_matches_naive_dilate_various_radii() {
        let data = make_random(50, 50, 67890);
        for r in [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0] {
            assert_bit_exact(MorphologyOperator::Dilate, r, r, &data, 50, 50);
        }
    }

    #[test]
    fn vhgw_matches_naive_asymmetric_radii() {
        let data = make_random(40, 30, 11111);
        assert_bit_exact(MorphologyOperator::Erode, 1.0, 10.0, &data, 40, 30);
        assert_bit_exact(MorphologyOperator::Erode, 10.0, 1.0, &data, 40, 30);
        assert_bit_exact(MorphologyOperator::Dilate, 1.0, 10.0, &data, 40, 30);
        assert_bit_exact(MorphologyOperator::Dilate, 10.0, 1.0, &data, 40, 30);
        assert_bit_exact(MorphologyOperator::Erode, 3.0, 7.0, &data, 40, 30);
        assert_bit_exact(MorphologyOperator::Dilate, 5.0, 2.0, &data, 40, 30);
    }

    #[test]
    fn vhgw_matches_naive_all_zeros() {
        let data = vec![RGBA8 { r: 0, g: 0, b: 0, a: 0 }; 20 * 20];
        assert_bit_exact(MorphologyOperator::Erode, 5.0, 5.0, &data, 20, 20);
        assert_bit_exact(MorphologyOperator::Dilate, 5.0, 5.0, &data, 20, 20);
    }

    #[test]
    fn vhgw_matches_naive_all_255() {
        let data = vec![RGBA8 { r: 255, g: 255, b: 255, a: 255 }; 20 * 20];
        assert_bit_exact(MorphologyOperator::Erode, 5.0, 5.0, &data, 20, 20);
        assert_bit_exact(MorphologyOperator::Dilate, 5.0, 5.0, &data, 20, 20);
    }

    #[test]
    fn vhgw_matches_naive_gradient() {
        let data = make_gradient(30, 30);
        assert_bit_exact(MorphologyOperator::Erode, 4.0, 4.0, &data, 30, 30);
        assert_bit_exact(MorphologyOperator::Dilate, 4.0, 4.0, &data, 30, 30);
    }

    #[test]
    fn vhgw_matches_naive_thin_images() {
        // 1×N
        let data = make_random(1, 30, 22222);
        assert_bit_exact(MorphologyOperator::Erode, 5.0, 5.0, &data, 1, 30);
        assert_bit_exact(MorphologyOperator::Dilate, 5.0, 5.0, &data, 1, 30);

        // N×1
        let data = make_random(30, 1, 33333);
        assert_bit_exact(MorphologyOperator::Erode, 5.0, 5.0, &data, 30, 1);
        assert_bit_exact(MorphologyOperator::Dilate, 5.0, 5.0, &data, 30, 1);
    }

    #[test]
    fn vhgw_matches_naive_radius_larger_than_dimension() {
        let data = make_random(10, 8, 44444);
        assert_bit_exact(MorphologyOperator::Erode, 20.0, 20.0, &data, 10, 8);
        assert_bit_exact(MorphologyOperator::Dilate, 20.0, 20.0, &data, 10, 8);
    }

    #[test]
    fn vhgw_matches_naive_fractional_radii() {
        let data = make_random(25, 25, 55555);
        assert_bit_exact(MorphologyOperator::Erode, 0.5, 0.5, &data, 25, 25);
        assert_bit_exact(MorphologyOperator::Dilate, 0.5, 0.5, &data, 25, 25);
        assert_bit_exact(MorphologyOperator::Erode, 2.3, 4.7, &data, 25, 25);
        assert_bit_exact(MorphologyOperator::Dilate, 1.1, 3.9, &data, 25, 25);
    }

    #[test]
    fn vhgw_matches_naive_small_image() {
        let data = make_random(3, 3, 66666);
        assert_bit_exact(MorphologyOperator::Erode, 2.0, 2.0, &data, 3, 3);
        assert_bit_exact(MorphologyOperator::Dilate, 2.0, 2.0, &data, 3, 3);
    }

}
