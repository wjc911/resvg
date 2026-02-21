// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Comprehensive benchmark for feComponentTransfer: tests LUT-optimized vs naive (direct)
//! paths across many image sizes, transfer function types, and input patterns.
//!
//! Special focus on the 256-pixel threshold where the code switches from direct
//! per-pixel computation to LUT-based lookup.
//!
//! Run with: cargo run -p resvg --release --example bench_component_transfer_comprehensive

use rgb::RGBA8;
use std::hint::black_box;
use std::time::{Duration, Instant};
use usvg::filter::{ComponentTransfer, Input, TransferFunction};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Number of iterations for timing. Higher = more stable.
const ITERATIONS: u32 = 100;

/// Warmup iterations before timing.
const WARMUP: u32 = 10;

/// Image sizes to test. Includes sizes around the 256-pixel threshold.
const IMAGE_SIZES: &[(u32, u32, &str)] = &[
    (1, 1, "1x1"),
    (4, 4, "4x4"),
    (15, 15, "15x15"),
    (16, 16, "16x16"),
    (64, 64, "64x64"),
    (15, 17, "15x17"),   // 255 pixels - just below threshold
    (16, 16, "16x16_t"), // 256 pixels - exact threshold
    (257, 1, "257x1"),   // 257 pixels - just above threshold
    (512, 512, "512x512"),
    (1024, 1024, "1024x1024"),
    (2048, 2048, "2048x2048"),
];

// ---------------------------------------------------------------------------
// Input pattern generators
// ---------------------------------------------------------------------------

fn make_zeros(count: usize) -> Vec<RGBA8> {
    vec![
        RGBA8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };
        count
    ]
}

fn make_opaque(count: usize) -> Vec<RGBA8> {
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

fn make_gradient(count: usize) -> Vec<RGBA8> {
    (0..count)
        .map(|i| {
            let t = if count > 1 {
                (i as f32 / (count - 1) as f32 * 255.0) as u8
            } else {
                128
            };
            RGBA8 {
                r: t,
                g: t,
                b: t,
                a: t,
            }
        })
        .collect()
}

fn make_random(count: usize) -> Vec<RGBA8> {
    // Deterministic pseudo-random using xorshift32
    let mut state: u32 = 0xDEAD_BEEF;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let bytes = state.to_le_bytes();
            RGBA8 {
                r: bytes[0],
                g: bytes[1],
                b: bytes[2],
                a: bytes[3],
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Naive (direct per-pixel) implementation -- verbatim copy of original code
// ---------------------------------------------------------------------------

fn f32_bound(min: f32, val: f32, max: f32) -> f32 {
    if val > max {
        max
    } else if val < min {
        min
    } else {
        val
    }
}

fn is_dummy(func: &TransferFunction) -> bool {
    match func {
        TransferFunction::Identity => true,
        TransferFunction::Table(values) => values.is_empty(),
        TransferFunction::Discrete(values) => values.is_empty(),
        TransferFunction::Linear { .. } => false,
        TransferFunction::Gamma { .. } => false,
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

fn apply_naive(fe: &ComponentTransfer, data: &mut [RGBA8]) {
    for pixel in data.iter_mut() {
        if !is_dummy(fe.func_r()) {
            pixel.r = transfer_scalar(fe.func_r(), pixel.r);
        }
        if !is_dummy(fe.func_b()) {
            pixel.b = transfer_scalar(fe.func_b(), pixel.b);
        }
        if !is_dummy(fe.func_g()) {
            pixel.g = transfer_scalar(fe.func_g(), pixel.g);
        }
        if !is_dummy(fe.func_a()) {
            pixel.a = transfer_scalar(fe.func_a(), pixel.a);
        }
    }
}

// ---------------------------------------------------------------------------
// LUT-optimized implementation -- mirrors the production code
// ---------------------------------------------------------------------------

fn apply_lut(fe: &ComponentTransfer, data: &mut [RGBA8]) {
    type Lut = [u8; 256];

    const IDENTITY_LUT: Lut = {
        let mut lut = [0u8; 256];
        let mut i = 0u16;
        while i < 256 {
            lut[i as usize] = i as u8;
            i += 1;
        }
        lut
    };

    fn build_lut(func: &TransferFunction) -> Lut {
        let mut lut: Lut = [0u8; 256];
        for i in 0u16..=255 {
            lut[i as usize] = transfer_scalar(func, i as u8);
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
        IDENTITY_LUT
    };
    let lut_g = if g_active {
        build_lut(fe.func_g())
    } else {
        IDENTITY_LUT
    };
    let lut_b = if b_active {
        build_lut(fe.func_b())
    } else {
        IDENTITY_LUT
    };
    let lut_a = if a_active {
        build_lut(fe.func_a())
    } else {
        IDENTITY_LUT
    };

    for pixel in data.iter_mut() {
        pixel.r = lut_r[pixel.r as usize];
        pixel.g = lut_g[pixel.g as usize];
        pixel.b = lut_b[pixel.b as usize];
        pixel.a = lut_a[pixel.a as usize];
    }
}

// ---------------------------------------------------------------------------
// Production-path implementation -- mirrors the updated per-function threshold
// ---------------------------------------------------------------------------

fn lut_threshold(func: &TransferFunction) -> usize {
    match func {
        TransferFunction::Identity => usize::MAX,
        TransferFunction::Gamma { .. } => 256,
        TransferFunction::Table(_) | TransferFunction::Discrete(_) => 1024,
        TransferFunction::Linear { .. } => 2048,
    }
}

fn apply_production(fe: &ComponentTransfer, data: &mut [RGBA8]) {
    type Lut = [u8; 256];
    const IDENTITY_LUT: Lut = {
        let mut lut = [0u8; 256];
        let mut i = 0u16;
        while i < 256 {
            lut[i as usize] = i as u8;
            i += 1;
        }
        lut
    };

    fn build_lut_inner(func: &TransferFunction) -> Lut {
        let mut lut: Lut = [0u8; 256];
        for i in 0u16..=255 {
            lut[i as usize] = transfer_scalar(func, i as u8);
        }
        lut
    }

    let func_r = fe.func_r();
    let func_g = fe.func_g();
    let func_b = fe.func_b();
    let func_a = fe.func_a();

    let r_active = !is_dummy(func_r);
    let g_active = !is_dummy(func_g);
    let b_active = !is_dummy(func_b);
    let a_active = !is_dummy(func_a);

    if !r_active && !g_active && !b_active && !a_active {
        return;
    }

    let pixel_count = data.len();

    let use_lut_r = r_active && pixel_count >= lut_threshold(func_r);
    let use_lut_g = g_active && pixel_count >= lut_threshold(func_g);
    let use_lut_b = b_active && pixel_count >= lut_threshold(func_b);
    let use_lut_a = a_active && pixel_count >= lut_threshold(func_a);

    let any_lut = use_lut_r || use_lut_g || use_lut_b || use_lut_a;

    if !any_lut {
        for pixel in data.iter_mut() {
            if r_active {
                pixel.r = transfer_scalar(func_r, pixel.r);
            }
            if g_active {
                pixel.g = transfer_scalar(func_g, pixel.g);
            }
            if b_active {
                pixel.b = transfer_scalar(func_b, pixel.b);
            }
            if a_active {
                pixel.a = transfer_scalar(func_a, pixel.a);
            }
        }
        return;
    }

    let lut_r = if use_lut_r {
        build_lut_inner(func_r)
    } else {
        IDENTITY_LUT
    };
    let lut_g = if use_lut_g {
        build_lut_inner(func_g)
    } else {
        IDENTITY_LUT
    };
    let lut_b = if use_lut_b {
        build_lut_inner(func_b)
    } else {
        IDENTITY_LUT
    };
    let lut_a = if use_lut_a {
        build_lut_inner(func_a)
    } else {
        IDENTITY_LUT
    };

    let any_direct = (r_active && !use_lut_r)
        || (g_active && !use_lut_g)
        || (b_active && !use_lut_b)
        || (a_active && !use_lut_a);

    if any_direct {
        for pixel in data.iter_mut() {
            pixel.r = if use_lut_r {
                lut_r[pixel.r as usize]
            } else if r_active {
                transfer_scalar(func_r, pixel.r)
            } else {
                pixel.r
            };
            pixel.g = if use_lut_g {
                lut_g[pixel.g as usize]
            } else if g_active {
                transfer_scalar(func_g, pixel.g)
            } else {
                pixel.g
            };
            pixel.b = if use_lut_b {
                lut_b[pixel.b as usize]
            } else if b_active {
                transfer_scalar(func_b, pixel.b)
            } else {
                pixel.b
            };
            pixel.a = if use_lut_a {
                lut_a[pixel.a as usize]
            } else if a_active {
                transfer_scalar(func_a, pixel.a)
            } else {
                pixel.a
            };
        }
    } else {
        for pixel in data.iter_mut() {
            pixel.r = lut_r[pixel.r as usize];
            pixel.g = lut_g[pixel.g as usize];
            pixel.b = lut_b[pixel.b as usize];
            pixel.a = lut_a[pixel.a as usize];
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_component_transfer_all(func: TransferFunction) -> ComponentTransfer {
    ComponentTransfer::new(
        Input::SourceGraphic,
        func.clone(),
        func.clone(),
        func.clone(),
        func,
    )
}

fn make_component_transfer_mixed() -> ComponentTransfer {
    ComponentTransfer::new(
        Input::SourceGraphic,
        TransferFunction::Gamma {
            amplitude: 1.0,
            exponent: 2.2,
            offset: 0.0,
        },
        TransferFunction::Linear {
            slope: 1.5,
            intercept: 0.1,
        },
        TransferFunction::Table(vec![0.0, 0.3, 0.6, 0.8, 1.0]),
        TransferFunction::Identity,
    )
}

#[derive(Clone, Copy)]
enum PathChoice {
    Naive,
    Lut,
    Production,
}

impl std::fmt::Display for PathChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathChoice::Naive => write!(f, "Direct"),
            PathChoice::Lut => write!(f, "LUT"),
            PathChoice::Production => write!(f, "Prod"),
        }
    }
}

struct BenchResult {
    size_label: String,
    pixels: u32,
    transfer_type: String,
    input_pattern: String,
    naive_us: f64,
    lut_us: f64,
    prod_us: f64,
    path_used: String,
}

fn bench_single(fe: &ComponentTransfer, base_image: &[RGBA8], path: PathChoice) -> Duration {
    // Warmup
    for _ in 0..WARMUP {
        let mut img = base_image.to_vec();
        match path {
            PathChoice::Naive => apply_naive(fe, &mut img),
            PathChoice::Lut => apply_lut(fe, &mut img),
            PathChoice::Production => apply_production(fe, &mut img),
        }
        black_box(&img);
    }

    // Timed
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let mut img = base_image.to_vec();
        match path {
            PathChoice::Naive => apply_naive(fe, &mut img),
            PathChoice::Lut => apply_lut(fe, &mut img),
            PathChoice::Production => apply_production(fe, &mut img),
        }
        black_box(&img);
    }
    start.elapsed() / ITERATIONS
}

/// Determines the production path label based on per-function thresholds.
fn prod_path_label(fe: &ComponentTransfer, pixel_count: usize) -> String {
    let funcs = [fe.func_r(), fe.func_g(), fe.func_b(), fe.func_a()];
    let any_lut = funcs
        .iter()
        .any(|f| !is_dummy(f) && pixel_count >= lut_threshold(f));
    let any_direct = funcs
        .iter()
        .any(|f| !is_dummy(f) && pixel_count < lut_threshold(f));
    if any_lut && any_direct {
        "Hybrid".to_string()
    } else if any_lut {
        "LUT".to_string()
    } else {
        "Direct".to_string()
    }
}

fn run_bench(
    fe: &ComponentTransfer,
    width: u32,
    height: u32,
    size_label: &str,
    transfer_type: &str,
    input_pattern: &str,
    base_image: &[RGBA8],
) -> BenchResult {
    let pixels = width * height;

    let naive_dur = bench_single(fe, base_image, PathChoice::Naive);
    let lut_dur = bench_single(fe, base_image, PathChoice::Lut);
    let prod_dur = bench_single(fe, base_image, PathChoice::Production);

    let path_used = prod_path_label(fe, pixels as usize);

    BenchResult {
        size_label: size_label.to_string(),
        pixels,
        transfer_type: transfer_type.to_string(),
        input_pattern: input_pattern.to_string(),
        naive_us: naive_dur.as_nanos() as f64 / 1000.0,
        lut_us: lut_dur.as_nanos() as f64 / 1000.0,
        prod_us: prod_dur.as_nanos() as f64 / 1000.0,
        path_used,
    }
}

// ---------------------------------------------------------------------------
// Transfer function definitions
// ---------------------------------------------------------------------------

struct TransferFuncDef {
    name: &'static str,
    func: TransferFunction,
}

fn transfer_functions() -> Vec<TransferFuncDef> {
    vec![
        TransferFuncDef {
            name: "Table(3)",
            func: TransferFunction::Table(vec![0.0, 0.5, 1.0]),
        },
        TransferFuncDef {
            name: "Table(7)",
            func: TransferFunction::Table(vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
        },
        TransferFuncDef {
            name: "Table(256)",
            func: TransferFunction::Table((0..256).map(|i| i as f32 / 255.0).collect()),
        },
        TransferFuncDef {
            name: "Discrete(4)",
            func: TransferFunction::Discrete(vec![0.0, 0.33, 0.67, 1.0]),
        },
        TransferFuncDef {
            name: "Discrete(8)",
            func: TransferFunction::Discrete(vec![0.0, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.0]),
        },
        TransferFuncDef {
            name: "Linear",
            func: TransferFunction::Linear {
                slope: 1.5,
                intercept: 0.1,
            },
        },
        TransferFuncDef {
            name: "Gamma(2.2)",
            func: TransferFunction::Gamma {
                amplitude: 1.0,
                exponent: 2.2,
                offset: 0.0,
            },
        },
        TransferFuncDef {
            name: "Gamma(0.45)",
            func: TransferFunction::Gamma {
                amplitude: 1.0,
                exponent: 0.4545,
                offset: 0.0,
            },
        },
    ]
}

fn input_patterns() -> Vec<(&'static str, Box<dyn Fn(usize) -> Vec<RGBA8>>)> {
    vec![
        ("zeros", Box::new(make_zeros)),
        ("opaque", Box::new(make_opaque)),
        ("gradient", Box::new(make_gradient)),
        ("random", Box::new(make_random)),
    ]
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("feComponentTransfer Comprehensive Benchmark");
    println!("============================================");
    println!("Iterations per measurement: {ITERATIONS}");
    println!("Warmup iterations: {WARMUP}");
    println!("Threshold: pixels < 256 -> Direct, >= 256 -> LUT");
    println!();

    let mut results: Vec<BenchResult> = Vec::new();
    let mut regressions: Vec<String> = Vec::new();

    let funcs = transfer_functions();
    let patterns = input_patterns();

    // Part 1: Comprehensive sweep with "opaque" input pattern for all transfer types & sizes
    println!("=== Part 1: All transfer types x all sizes (opaque input) ===");
    println!();

    for func_def in &funcs {
        let fe = make_component_transfer_all(func_def.func.clone());
        for &(w, h, label) in IMAGE_SIZES {
            let pixels = (w * h) as usize;
            let base_image = make_opaque(pixels);
            let r = run_bench(&fe, w, h, label, func_def.name, "opaque", &base_image);
            results.push(r);
        }
    }

    // Part 2: Mixed channel test (R=gamma, G=linear, B=table, A=identity)
    println!("=== Part 2: Mixed channels (R=Gamma, G=Linear, B=Table, A=Identity) ===");
    println!();

    let fe_mixed = make_component_transfer_mixed();
    for &(w, h, label) in IMAGE_SIZES {
        let pixels = (w * h) as usize;
        let base_image = make_opaque(pixels);
        let r = run_bench(&fe_mixed, w, h, label, "Mixed", "opaque", &base_image);
        results.push(r);
    }

    // Part 3: Input pattern variations for Gamma(2.2) at key sizes
    println!("=== Part 3: Input pattern variations for Gamma(2.2) ===");
    println!();

    let fe_gamma = make_component_transfer_all(TransferFunction::Gamma {
        amplitude: 1.0,
        exponent: 2.2,
        offset: 0.0,
    });

    let key_sizes: &[(u32, u32, &str)] = &[
        (15, 17, "15x17"),   // 255 pixels
        (16, 16, "16x16_t"), // 256 pixels
        (512, 512, "512x512"),
        (1024, 1024, "1024x1024"),
    ];

    for &(w, h, label) in key_sizes {
        let pixels = (w * h) as usize;
        for (pat_name, pat_fn) in &patterns {
            let base_image = pat_fn(pixels);
            let r = run_bench(&fe_gamma, w, h, label, "Gamma(2.2)", pat_name, &base_image);
            results.push(r);
        }
    }

    // Part 4: Input pattern variations for Linear at key sizes
    let fe_linear = make_component_transfer_all(TransferFunction::Linear {
        slope: 1.5,
        intercept: 0.1,
    });

    for &(w, h, label) in key_sizes {
        let pixels = (w * h) as usize;
        for (pat_name, pat_fn) in &patterns {
            let base_image = pat_fn(pixels);
            let r = run_bench(&fe_linear, w, h, label, "Linear", pat_name, &base_image);
            results.push(r);
        }
    }

    // ---------------------------------------------------------------------------
    // Print results table
    // ---------------------------------------------------------------------------
    println!();
    println!(
        "{:<12} | {:<14} | {:<10} | {:>12} | {:>12} | {:>8} | {:<8} | {:>10}",
        "Image Size",
        "Transfer Type",
        "Input",
        "Direct (us)",
        "LUT (us)",
        "Speedup",
        "Path",
        "Prod (us)"
    );
    println!("{}", "-".repeat(108));

    for r in &results {
        let speedup = r.naive_us / r.lut_us;
        // Check if production path is slower than the best of naive/LUT by >5%
        let best = r.naive_us.min(r.lut_us);
        let prod_regression = r.prod_us > best * 1.05;
        let flag = if prod_regression {
            " *** REGRESSION (prod vs best)"
        } else {
            ""
        };

        println!(
            "{:<12} | {:<14} | {:<10} | {:>12.2} | {:>12.2} | {:>7.2}x | {:<8} | {:>10.2}{}",
            format!("{} ({}px)", r.size_label, r.pixels),
            r.transfer_type,
            r.input_pattern,
            r.naive_us,
            r.lut_us,
            speedup,
            r.path_used,
            r.prod_us,
            flag
        );

        // Check for regressions (>5%): production path slower than best alternative
        if prod_regression {
            let pct = (r.prod_us / best - 1.0) * 100.0;
            regressions.push(format!(
                "REGRESSION: {} {} {} - Prod ({}) is {:.1}% slower than best ({:.2} us vs {:.2} us)",
                r.size_label,
                r.transfer_type,
                r.input_pattern,
                r.path_used,
                pct,
                r.prod_us,
                best,
            ));
        }
    }

    // ---------------------------------------------------------------------------
    // Threshold analysis
    // ---------------------------------------------------------------------------
    println!();
    println!("=== Threshold Analysis (per-function thresholds) ===");
    println!("  Gamma: 256px, Table/Discrete: 1024px, Linear: 2048px");
    println!();

    // Collect threshold-relevant results
    let threshold_sizes = ["15x17", "16x16_t", "257x1", "64x64"];
    for r in &results {
        if threshold_sizes.contains(&r.size_label.as_str()) {
            let speedup = r.naive_us / r.lut_us;
            let best = r.naive_us.min(r.lut_us);
            let prod_optimal = r.prod_us <= best * 1.05;
            println!(
                "  {} ({} px) | {:<14} | Direct={:.2}us LUT={:.2}us Prod={:.2}us | LUT speedup={:.2}x | Path={} | Optimal: {}",
                r.size_label,
                r.pixels,
                r.transfer_type,
                r.naive_us,
                r.lut_us,
                r.prod_us,
                speedup,
                r.path_used,
                if prod_optimal { "YES" } else { "NO" }
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------------------
    println!();
    println!("=== Summary ===");
    println!();

    if regressions.is_empty() {
        println!("No regressions found (>5% threshold).");
    } else {
        println!("REGRESSIONS FOUND ({}):", regressions.len());
        for r in &regressions {
            println!("  - {}", r);
        }
    }

    println!();
    println!("Total benchmarks run: {}", results.len());
    println!("Done.");
}
