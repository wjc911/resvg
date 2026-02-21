// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for feComposite Arithmetic mode.
//!
//! Compares the naive (original) and SoA-batched optimized implementations,
//! verifies bit-exact output, and reports throughput at multiple resolutions.
//!
//! Run with: cargo bench --bench composite_arithmetic -p resvg

use std::hint::black_box;
use std::time::Instant;

use rgb::RGBA8;

// ---------------------------------------------------------------------------
// Local copies of helpers (avoids depending on private internals)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct ImageRef<'a> {
    data: &'a [RGBA8],
    width: u32,
    height: u32,
}

#[allow(dead_code)]
struct ImageRefMut<'a> {
    data: &'a mut [RGBA8],
    width: u32,
    height: u32,
}

#[inline]
fn f32_bound(min: f32, val: f32, max: f32) -> f32 {
    if val > max {
        max
    } else if val < min {
        min
    } else {
        val
    }
}

/// ULP-based approximate zero check matching `ApproxZeroUlps` for f32.
#[inline]
fn approx_zero_ulps_f32(val: f32, ulps: i32) -> bool {
    let a_bits = val.to_bits() as i32;
    let b_bits = 0.0f32.to_bits() as i32;
    (a_bits - b_bits).abs() <= ulps
}

// ---------------------------------------------------------------------------
// Naive implementation — verbatim copy of the original
// ---------------------------------------------------------------------------

fn arithmetic_naive(
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
    src1: &ImageRef,
    src2: &ImageRef,
    dest: &mut ImageRefMut,
) {
    let calc = |i1: u8, i2: u8, max: f32| {
        let i1 = i1 as f32 / 255.0;
        let i2 = i2 as f32 / 255.0;
        let result = k1 * i1 * i2 + k2 * i1 + k3 * i2 + k4;
        f32_bound(0.0, result, max)
    };

    let mut i = 0;
    for (c1, c2) in src1.data.iter().zip(src2.data.iter()) {
        let a = calc(c1.a, c2.a, 1.0);
        if approx_zero_ulps_f32(a, 4) {
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

// ---------------------------------------------------------------------------
// Optimized implementation — SoA batched processing
// ---------------------------------------------------------------------------

const BATCH: usize = 64;

#[inline(always)]
fn clamp_and_scale(val: f32, max: f32) -> u8 {
    let clamped = if val > max {
        max
    } else if val < 0.0 {
        0.0
    } else {
        val
    };
    (clamped * 255.0) as u8
}

fn arithmetic_optimized(
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
    src1: &ImageRef,
    src2: &ImageRef,
    dest: &mut ImageRefMut,
) {
    let len = src1.data.len();
    let mut offset = 0;
    const INV_255: f32 = 1.0 / 255.0;

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

        // AoS -> SoA conversion + normalize
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

        // Pure math — no branches, vectorizable
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

        // Clamp alpha
        for j in 0..batch_len {
            if res_a[j] > 1.0 {
                res_a[j] = 1.0;
            } else if res_a[j] < 0.0 {
                res_a[j] = 0.0;
            }
        }

        // Write back
        let dest_slice = &mut dest.data[offset..offset + batch_len];
        for j in 0..batch_len {
            let a = res_a[j];
            if approx_zero_ulps_f32(a, 4) {
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

// ---------------------------------------------------------------------------
// Test data generation
// ---------------------------------------------------------------------------

fn generate_test_data(width: u32, height: u32, seed: u8) -> Vec<RGBA8> {
    let len = (width * height) as usize;
    let mut data = Vec::with_capacity(len);
    for i in 0..len {
        let v = ((i as u32).wrapping_mul(seed as u32 + 1).wrapping_add(17)) as u8;
        let r = v.wrapping_add(seed);
        let g = v.wrapping_add(seed.wrapping_mul(2));
        let b = v.wrapping_add(seed.wrapping_mul(3));
        let a = v.wrapping_add(seed.wrapping_mul(5));
        data.push(RGBA8 { r, g, b, a });
    }
    data
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

fn bench_one(
    label: &str,
    width: u32,
    height: u32,
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
    iterations: u32,
) {
    let src1_data = generate_test_data(width, height, 37);
    let src2_data = generate_test_data(width, height, 101);
    let pixel_count = (width * height) as usize;
    let zero = RGBA8 {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    let mut dest_naive = vec![zero; pixel_count];
    let mut dest_opt = vec![zero; pixel_count];

    let src1 = ImageRef {
        data: &src1_data,
        width,
        height,
    };
    let src2 = ImageRef {
        data: &src2_data,
        width,
        height,
    };

    // --- Correctness: verify bit-exact match ---
    {
        let mut dn = ImageRefMut {
            data: &mut dest_naive,
            width,
            height,
        };
        arithmetic_naive(k1, k2, k3, k4, &src1, &src2, &mut dn);
    }
    {
        let mut do_ = ImageRefMut {
            data: &mut dest_opt,
            width,
            height,
        };
        arithmetic_optimized(k1, k2, k3, k4, &src1, &src2, &mut do_);
    }

    let mut mismatch_count = 0;
    for idx in 0..pixel_count {
        if dest_naive[idx] != dest_opt[idx] {
            if mismatch_count < 5 {
                eprintln!(
                    "  MISMATCH at pixel {}: naive={:?}  opt={:?}",
                    idx, dest_naive[idx], dest_opt[idx]
                );
            }
            mismatch_count += 1;
        }
    }
    if mismatch_count > 0 {
        eprintln!("  ** {} total mismatches for {} **", mismatch_count, label);
    }

    // --- Benchmark: naive ---
    dest_naive.fill(zero);
    for _ in 0..2 {
        let mut d = ImageRefMut {
            data: &mut dest_naive,
            width,
            height,
        };
        arithmetic_naive(k1, k2, k3, k4, &src1, &src2, &mut d);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let mut d = ImageRefMut {
            data: &mut dest_naive,
            width,
            height,
        };
        arithmetic_naive(
            black_box(k1),
            black_box(k2),
            black_box(k3),
            black_box(k4),
            black_box(&src1),
            black_box(&src2),
            black_box(&mut d),
        );
    }
    let elapsed_naive = start.elapsed();

    // --- Benchmark: optimized ---
    dest_opt.fill(zero);
    for _ in 0..2 {
        let mut d = ImageRefMut {
            data: &mut dest_opt,
            width,
            height,
        };
        arithmetic_optimized(k1, k2, k3, k4, &src1, &src2, &mut d);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let mut d = ImageRefMut {
            data: &mut dest_opt,
            width,
            height,
        };
        arithmetic_optimized(
            black_box(k1),
            black_box(k2),
            black_box(k3),
            black_box(k4),
            black_box(&src1),
            black_box(&src2),
            black_box(&mut d),
        );
    }
    let elapsed_opt = start.elapsed();

    let total_pixels = pixel_count as u64 * iterations as u64;
    let mpx_naive = total_pixels as f64 / elapsed_naive.as_secs_f64() / 1_000_000.0;
    let mpx_opt = total_pixels as f64 / elapsed_opt.as_secs_f64() / 1_000_000.0;
    let speedup = mpx_opt / mpx_naive;

    println!(
        "  {:<40}  naive {:>7.1} Mpx/s | opt {:>7.1} Mpx/s | speedup {:>5.2}x{}",
        label,
        mpx_naive,
        mpx_opt,
        speedup,
        if mismatch_count > 0 {
            "  ** MISMATCH **"
        } else {
            ""
        },
    );
}

fn main() {
    println!("feComposite Arithmetic Benchmark — Naive vs SoA-Batched Optimized");
    println!("=================================================================");
    println!();

    let resolutions: &[(u32, u32, &str)] = &[
        (64, 64, "64x64"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
        (1024, 1024, "1024x1024"),
        (2048, 2048, "2048x2048"),
    ];

    let k_values: &[(f32, f32, f32, f32, &str)] = &[
        (0.0, 1.0, 0.0, 0.0, "k=(0,1,0,0) passthrough"),
        (0.0, 0.0, 1.0, 0.0, "k=(0,0,1,0) passthrough"),
        (1.0, 0.0, 0.0, 0.0, "k=(1,0,0,0) multiply"),
        (0.5, 0.5, 0.5, 0.0, "k=(0.5,0.5,0.5,0) blend"),
        (1.0, 1.0, 1.0, -0.5, "k=(1,1,1,-0.5) full arith"),
        (0.0, 0.0, 0.0, 0.0, "k=(0,0,0,0) zeros"),
    ];

    for &(w, h, res_label) in resolutions {
        let pixel_count = w * h;
        let iters = (10_000_000u32 / pixel_count).max(5);

        println!(
            "Resolution: {} ({} pixels, {} iterations)",
            res_label, pixel_count, iters
        );

        for &(k1, k2, k3, k4, k_label) in k_values {
            bench_one(k_label, w, h, k1, k2, k3, k4, iters);
        }
        println!();
    }
}
