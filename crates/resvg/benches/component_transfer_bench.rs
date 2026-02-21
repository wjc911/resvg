// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for feComponentTransfer: compares LUT-optimized vs naive (per-pixel) paths.
//!
//! Run with: cargo bench -p resvg --bench component_transfer_bench

use rgb::RGBA8;
use std::hint::black_box;
use std::time::{Duration, Instant};
use usvg::filter::{ComponentTransfer, Input, TransferFunction};

/// Number of iterations per benchmark run for stable timing.
const ITERATIONS: u32 = 50;

/// Image resolutions to benchmark.
const RESOLUTIONS: &[(u32, u32)] = &[
    (64, 64),     // 4K pixels
    (256, 256),   // 64K pixels
    (512, 512),   // 256K pixels
    (1024, 1024), // 1M pixels
    (2048, 2048), // 4M pixels
];

fn make_test_image(width: u32, height: u32) -> Vec<RGBA8> {
    let count = (width * height) as usize;
    (0..count)
        .map(|i| {
            let v = (i % 256) as u8;
            RGBA8 {
                r: v,
                g: v.wrapping_add(85),
                b: v.wrapping_add(170),
                a: 255,
            }
        })
        .collect()
}

fn make_component_transfer(func: TransferFunction) -> ComponentTransfer {
    ComponentTransfer::new(
        Input::SourceGraphic,
        func.clone(),
        func.clone(),
        func.clone(),
        func,
    )
}

/// The naive per-pixel implementation (original code, verbatim).
fn apply_naive(fe: &ComponentTransfer, data: &mut [RGBA8]) {
    fn is_dummy(func: &TransferFunction) -> bool {
        match func {
            TransferFunction::Identity => true,
            TransferFunction::Table(values) => values.is_empty(),
            TransferFunction::Discrete(values) => values.is_empty(),
            TransferFunction::Linear { .. } => false,
            TransferFunction::Gamma { .. } => false,
        }
    }

    fn f32_bound(min: f32, val: f32, max: f32) -> f32 {
        if val > max {
            max
        } else if val < min {
            min
        } else {
            val
        }
    }

    fn transfer(func: &TransferFunction, c: u8) -> u8 {
        let c = c as f32 / 255.0;
        let c = match func {
            TransferFunction::Identity => c,
            TransferFunction::Table(values) => {
                let n = values.len() - 1;
                let k = (c * (n as f32)).floor() as usize;
                let k = std::cmp::min(k, n);
                if k == n {
                    values[k]
                } else {
                    let vk = values[k];
                    let vk1 = values[k + 1];
                    let k = k as f32;
                    let n = n as f32;
                    vk + (c - k / n) * n * (vk1 - vk)
                }
            }
            TransferFunction::Discrete(values) => {
                let n = values.len();
                let k = (c * (n as f32)).floor() as usize;
                values[std::cmp::min(k, n - 1)]
            }
            TransferFunction::Linear { slope, intercept } => slope * c + intercept,
            TransferFunction::Gamma {
                amplitude,
                exponent,
                offset,
            } => amplitude * c.powf(*exponent) + offset,
        };

        (f32_bound(0.0, c, 1.0) * 255.0) as u8
    }

    for pixel in data.iter_mut() {
        if !is_dummy(fe.func_r()) {
            pixel.r = transfer(fe.func_r(), pixel.r);
        }
        if !is_dummy(fe.func_b()) {
            pixel.b = transfer(fe.func_b(), pixel.b);
        }
        if !is_dummy(fe.func_g()) {
            pixel.g = transfer(fe.func_g(), pixel.g);
        }
        if !is_dummy(fe.func_a()) {
            pixel.a = transfer(fe.func_a(), pixel.a);
        }
    }
}

/// The LUT-optimized implementation (mirrors the production code).
fn apply_lut(fe: &ComponentTransfer, data: &mut [RGBA8]) {
    type Lut = [u8; 256];

    fn is_dummy(func: &TransferFunction) -> bool {
        match func {
            TransferFunction::Identity => true,
            TransferFunction::Table(values) => values.is_empty(),
            TransferFunction::Discrete(values) => values.is_empty(),
            TransferFunction::Linear { .. } => false,
            TransferFunction::Gamma { .. } => false,
        }
    }

    fn f32_bound(min: f32, val: f32, max: f32) -> f32 {
        if val > max {
            max
        } else if val < min {
            min
        } else {
            val
        }
    }

    fn transfer_scalar(func: &TransferFunction, c: u8) -> u8 {
        let c = c as f32 / 255.0;
        let c = match func {
            TransferFunction::Identity => c,
            TransferFunction::Table(values) => {
                let n = values.len() - 1;
                let k = (c * (n as f32)).floor() as usize;
                let k = std::cmp::min(k, n);
                if k == n {
                    values[k]
                } else {
                    let vk = values[k];
                    let vk1 = values[k + 1];
                    let k = k as f32;
                    let n = n as f32;
                    vk + (c - k / n) * n * (vk1 - vk)
                }
            }
            TransferFunction::Discrete(values) => {
                let n = values.len();
                let k = (c * (n as f32)).floor() as usize;
                values[std::cmp::min(k, n - 1)]
            }
            TransferFunction::Linear { slope, intercept } => slope * c + intercept,
            TransferFunction::Gamma {
                amplitude,
                exponent,
                offset,
            } => amplitude * c.powf(*exponent) + offset,
        };

        (f32_bound(0.0, c, 1.0) * 255.0) as u8
    }

    fn build_lut(func: &TransferFunction) -> Lut {
        let mut lut: Lut = [0u8; 256];
        for i in 0u16..=255 {
            lut[i as usize] = transfer_scalar(func, i as u8);
        }
        lut
    }

    fn identity_lut() -> Lut {
        let mut lut: Lut = [0u8; 256];
        let mut i = 0u16;
        while i <= 255 {
            lut[i as usize] = i as u8;
            i += 1;
        }
        lut
    }

    let r_active = !is_dummy(fe.func_r());
    let g_active = !is_dummy(fe.func_g());
    let b_active = !is_dummy(fe.func_b());
    let a_active = !is_dummy(fe.func_a());

    if !r_active && !g_active && !b_active && !a_active {
        return;
    }

    let lut_r = if r_active {
        build_lut(fe.func_r())
    } else {
        identity_lut()
    };
    let lut_g = if g_active {
        build_lut(fe.func_g())
    } else {
        identity_lut()
    };
    let lut_b = if b_active {
        build_lut(fe.func_b())
    } else {
        identity_lut()
    };
    let lut_a = if a_active {
        build_lut(fe.func_a())
    } else {
        identity_lut()
    };

    for pixel in data.iter_mut() {
        pixel.r = lut_r[pixel.r as usize];
        pixel.g = lut_g[pixel.g as usize];
        pixel.b = lut_b[pixel.b as usize];
        pixel.a = lut_a[pixel.a as usize];
    }
}

fn bench_one(
    name: &str,
    fe: &ComponentTransfer,
    width: u32,
    height: u32,
    f: impl Fn(&ComponentTransfer, &mut [RGBA8]),
) -> Duration {
    let base_image = make_test_image(width, height);
    // Warmup
    for _ in 0..3 {
        let mut img = base_image.clone();
        f(fe, &mut img);
        black_box(&img);
    }
    // Timed
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let mut img = base_image.clone();
        f(fe, &mut img);
        black_box(&img);
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / ITERATIONS;
    let pixels = (width as u64) * (height as u64);
    let mpps = (pixels as f64) / per_iter.as_secs_f64() / 1_000_000.0;
    println!(
        "  {name:<12} {width}x{height} ({pixels:>8} px): {per_iter:>10.2?}/iter  ({mpps:.1} Mpx/s)"
    );
    per_iter
}

fn bench_transfer_type(name: &str, func: TransferFunction) {
    println!("\n=== {name} ===");
    let fe = make_component_transfer(func);

    for &(w, h) in RESOLUTIONS {
        let naive_time = bench_one("naive", &fe, w, h, |fe, data| apply_naive(fe, data));
        let lut_time = bench_one("lut", &fe, w, h, |fe, data| apply_lut(fe, data));

        let speedup = naive_time.as_secs_f64() / lut_time.as_secs_f64();
        println!("  speedup: {speedup:.2}x\n");
    }
}

fn main() {
    println!("feComponentTransfer Benchmark: LUT-optimized vs Naive (per-pixel)");
    println!("Iterations per measurement: {ITERATIONS}");
    println!("================================================================");

    bench_transfer_type(
        "Gamma (amplitude=1.0, exponent=2.2, offset=0.0)",
        TransferFunction::Gamma {
            amplitude: 1.0,
            exponent: 2.2,
            offset: 0.0,
        },
    );

    bench_transfer_type(
        "Gamma (amplitude=0.8, exponent=0.45, offset=0.1)",
        TransferFunction::Gamma {
            amplitude: 0.8,
            exponent: 0.45,
            offset: 0.1,
        },
    );

    bench_transfer_type(
        "Linear (slope=0.5, intercept=0.25)",
        TransferFunction::Linear {
            slope: 0.5,
            intercept: 0.25,
        },
    );

    bench_transfer_type(
        "Table (3 entries)",
        TransferFunction::Table(vec![0.0, 0.5, 1.0]),
    );

    bench_transfer_type(
        "Table (7 entries)",
        TransferFunction::Table(vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
    );

    bench_transfer_type(
        "Discrete (4 entries)",
        TransferFunction::Discrete(vec![0.0, 0.33, 0.67, 1.0]),
    );

    println!("\n================================================================");
    println!("Done.");
}
