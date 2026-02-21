// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{ImageRef, ImageRefMut, f32_bound};
use rgb::RGBA8;
use usvg::ApproxZeroUlps;

/// Crossover pixel count below which the naive implementation is used.
/// For very small images, the overhead of the optimized path is not worthwhile.
const OPTIMIZED_CROSSOVER: usize = 64;

/// Batch size for the optimized path. Must be large enough to amortize
/// per-batch overhead and give LLVM a chance to auto-vectorize the
/// arithmetic loops, but small enough to stay in L1 cache.
/// 64 pixels * 4 channels * 4 bytes = 1 KiB per buffer, well within L1.
const BATCH: usize = 64;

/// Original (naive) arithmetic implementation — preserved verbatim for correctness
/// reference and for small images where setup overhead dominates.
fn arithmetic_naive(
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
    src1: ImageRef,
    src2: ImageRef,
    dest: ImageRefMut,
) {
    let calc = |i1, i2, max| {
        let i1 = i1 as f32 / 255.0;
        let i2 = i2 as f32 / 255.0;
        let result = k1 * i1 * i2 + k2 * i1 + k3 * i2 + k4;
        f32_bound(0.0, result, max)
    };

    let mut i = 0;
    for (c1, c2) in src1.data.iter().zip(src2.data.iter()) {
        let a = calc(c1.a, c2.a, 1.0);
        if a.approx_zero_ulps(4) {
            i += 1;
            continue;
        }

        let r = (calc(c1.r, c2.r, a) * 255.0) as u8;
        let g = (calc(c1.g, c2.g, a) * 255.0) as u8;
        let b = (calc(c1.b, c2.b, a) * 255.0) as u8;
        let a = (a * 255.0) as u8;

        dest.data[i] = RGBA8 { r, g, b, a };

        i += 1;
    }
}

/// Optimized arithmetic composition using SoA batch processing.
///
/// Processes pixels in fixed-size batches using separate `[f32; BATCH]` arrays
/// per channel (Structure-of-Arrays layout). The SoA layout enables potential
/// LLVM auto-vectorization of the arithmetic loops (Step 2), though the
/// AoS-to-SoA conversion (Step 1), SoA-to-AoS write-back (Step 4), and the
/// per-pixel alpha-zero branch are inherently scalar. The main benefit is
/// better instruction-level parallelism and cache locality for the arithmetic
/// computation.
fn arithmetic_optimized(
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
    src1: ImageRef,
    src2: ImageRef,
    dest: ImageRefMut,
) {
    let len = src1.data.len();
    let mut offset = 0;

    // Precompute reciprocal to avoid per-pixel division.
    const INV_255: f32 = 1.0 / 255.0;

    // Scratch buffers for batch processing — one per channel (SoA layout).
    // Separating channels into contiguous arrays allows the arithmetic
    // loops (Step 2) to be candidates for LLVM auto-vectorization.
    let mut r1 = [0.0f32; BATCH];
    let mut g1 = [0.0f32; BATCH];
    let mut b1 = [0.0f32; BATCH];
    let mut a1 = [0.0f32; BATCH];
    let mut r2 = [0.0f32; BATCH];
    let mut g2 = [0.0f32; BATCH];
    let mut b2 = [0.0f32; BATCH];
    let mut a2 = [0.0f32; BATCH];

    let mut res_r = [0.0f32; BATCH];
    let mut res_g = [0.0f32; BATCH];
    let mut res_b = [0.0f32; BATCH];
    let mut res_a = [0.0f32; BATCH];

    while offset < len {
        let batch_len = (len - offset).min(BATCH);

        let s1 = &src1.data[offset..offset + batch_len];
        let s2 = &src2.data[offset..offset + batch_len];

        // Step 1: Convert AoS u8 pixels to SoA f32 channels (normalized).
        // This scatter/gather loop is inherently scalar due to the AoS input
        // layout, but is simple and predictable for the CPU's load pipeline.
        for j in 0..batch_len {
            r1[j] = s1[j].r as f32 * INV_255;
            g1[j] = s1[j].g as f32 * INV_255;
            b1[j] = s1[j].b as f32 * INV_255;
            a1[j] = s1[j].a as f32 * INV_255;

            r2[j] = s2[j].r as f32 * INV_255;
            g2[j] = s2[j].g as f32 * INV_255;
            b2[j] = s2[j].b as f32 * INV_255;
            a2[j] = s2[j].a as f32 * INV_255;
        }

        // Step 2: Apply arithmetic formula per channel — no branches, pure math.
        // These branchless loops over contiguous f32 arrays are candidates for
        // LLVM auto-vectorization (the SoA layout is what makes this possible).
        for j in 0..batch_len {
            res_r[j] = k1 * r1[j] * r2[j] + k2 * r1[j] + k3 * r2[j] + k4;
        }
        for j in 0..batch_len {
            res_g[j] = k1 * g1[j] * g2[j] + k2 * g1[j] + k3 * g2[j] + k4;
        }
        for j in 0..batch_len {
            res_b[j] = k1 * b1[j] * b2[j] + k2 * b1[j] + k3 * b2[j] + k4;
        }
        for j in 0..batch_len {
            res_a[j] = k1 * a1[j] * a2[j] + k2 * a1[j] + k3 * a2[j] + k4;
        }

        // Step 3: Clamp alpha to [0, 1].
        for j in 0..batch_len {
            res_a[j] = res_a[j].clamp(0.0, 1.0);
        }

        // Step 4: Write results back (SoA -> AoS), with per-pixel alpha-zero
        // check. This loop is inherently scalar due to the conditional branch
        // and the interleaved AoS store.
        let dest_slice = &mut dest.data[offset..offset + batch_len];
        for j in 0..batch_len {
            let a = res_a[j];
            if a.approx_zero_ulps(4) {
                continue;
            }

            let r = clamp_and_scale(res_r[j], a);
            let g = clamp_and_scale(res_g[j], a);
            let b = clamp_and_scale(res_b[j], a);
            let a = (a * 255.0) as u8;

            dest_slice[j] = RGBA8 { r, g, b, a };
        }

        offset += batch_len;
    }
}

/// Clamp `val` to `[0, max]` and scale to `[0, 255]` as `u8`.
#[inline(always)]
fn clamp_and_scale(val: f32, max: f32) -> u8 {
    (val.clamp(0.0, max) * 255.0) as u8
}

/// Performs an arithmetic composition.
///
/// - `src1` and `src2` image pixels should have a **premultiplied alpha**.
/// - `dest` image pixels will have a **premultiplied alpha**.
///
/// # Panics
///
/// When `src1`, `src2` and `dest` have different sizes.
pub fn arithmetic(
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
    src1: ImageRef,
    src2: ImageRef,
    dest: ImageRefMut,
) {
    assert!(src1.width == src2.width && src1.width == dest.width);
    assert!(src1.height == src2.height && src1.height == dest.height);

    // Fast path for degenerate k-values — checked before the pixel-count
    // crossover so we never do unnecessary batch work.

    // All coefficients zero: output is always zero regardless of input.
    if k1 == 0.0 && k2 == 0.0 && k3 == 0.0 && k4 == 0.0 {
        // dest is already zero-initialized by the caller, nothing to do.
        return;
    }

    // When k4==0 and at least one of k1/k2/k3 is also zero, many input
    // combinations produce zero output alpha. The naive path's per-pixel
    // early-exit on alpha~=0 is optimal here — it skips all channel math
    // for transparent pixels, which the batched path cannot do.
    if k4 == 0.0 && (k1 == 0.0 || k2 == 0.0 || k3 == 0.0) {
        arithmetic_naive(k1, k2, k3, k4, src1, src2, dest);
        return;
    }

    let pixel_count = src1.data.len();
    if pixel_count < OPTIMIZED_CROSSOVER {
        arithmetic_naive(k1, k2, k3, k4, src1, src2, dest);
    } else {
        arithmetic_optimized(k1, k2, k3, k4, src1, src2, dest);
    }
}
