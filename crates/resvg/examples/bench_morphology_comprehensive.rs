// Comprehensive performance benchmark for feMorphology filter optimization.
//
// Scenarios modelled after real-world SVG morphology usage:
//   - Icon text outline (dilate r=2, small images)
//   - Heading outline (dilate r=3-4, medium images)
//   - Subtle erode (erode r=1, various sizes)
//   - Thick knockout (dilate r=8, stress test for vHGW)
//   - Threshold boundary (radii near NAIVE_KERNEL_AREA_THRESHOLD=9)
//   - Non-square images (real aspect ratios like 200x150, 400x300)
//   - Asymmetric radius (directional effects: rx!=ry)
//
// Input patterns:
//   - "opaque": solid RGBA with random RGB, alpha=255 (typical rendered content)
//   - "text-like": sparse alpha simulating text glyph maps (bright on transparent)
//
// Uses multithreading via std::thread::scope for faster execution.
//
// Run with:
//   cargo run --release --example bench_morphology_comprehensive

use resvg::filter::ImageRefMut;
use resvg::filter::morphology;
use rgb::RGBA8;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use usvg::filter::MorphologyOperator;

// Mirror the threshold from morphology.rs
const NAIVE_KERNEL_AREA_THRESHOLD: u32 = 9;

// ---------------------------------------------------------------------------
// Configuration struct for one benchmark run
// ---------------------------------------------------------------------------

struct Config {
    order: usize,
    scenario: &'static str,
    width: u32,
    height: u32,
    rx: f32,
    ry: f32,
    operator: MorphologyOperator,
    pattern: &'static str,
    iterations: usize,
}

// ---------------------------------------------------------------------------
// Image generators
// ---------------------------------------------------------------------------

/// Solid RGBA pixels with random RGB values, alpha=255.
/// Represents typical fully-rendered content (icons, backgrounds).
fn make_opaque(w: u32, h: u32, seed: u64) -> Vec<RGBA8> {
    let mut data = Vec::with_capacity((w * h) as usize);
    let mut state = seed;
    for _ in 0..(w * h) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bytes = state.to_le_bytes();
        data.push(RGBA8 {
            r: bytes[0],
            g: bytes[1],
            b: bytes[2],
            a: 255,
        });
    }
    data
}

/// Sparse alpha simulating text glyph alpha maps: ~30% of pixels have bright
/// foreground (high alpha, white-ish RGB), the rest are fully transparent.
/// This matches how SVG text outlines look before morphology is applied.
fn make_text_like(w: u32, h: u32, seed: u64) -> Vec<RGBA8> {
    let mut data = Vec::with_capacity((w * h) as usize);
    let mut state = seed;
    for _ in 0..(w * h) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bytes = state.to_le_bytes();
        // ~30% coverage: text glyphs are sparse
        let is_glyph = bytes[4] < 77; // 77/256 ~ 30%
        if is_glyph {
            // Bright foreground pixel (premultiplied alpha)
            let a = 200u8.saturating_add(bytes[5] & 0x3F); // alpha 200-255
            data.push(RGBA8 {
                r: a,
                g: a,
                b: a,
                a,
            });
        } else {
            data.push(RGBA8 {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            });
        }
    }
    data
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

fn median(values: &mut [Duration]) -> Duration {
    values.sort();
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2
    } else {
        values[n / 2]
    }
}

fn iterations_for_config(w: u32, h: u32, rx: f32, ry: f32) -> usize {
    let pixels = (w as u64) * (h as u64);
    let columns = std::cmp::min(rx.ceil() as u64 * 2, w as u64);
    let rows = std::cmp::min(ry.ceil() as u64 * 2, h as u64);
    let kernel_area = columns * rows;
    let naive_work = pixels * kernel_area;

    // Target: each config completes within ~0.5s for naive path.
    let base = if pixels <= 1024 {
        10_000
    } else if pixels <= 65536 {
        1_000
    } else {
        100
    };

    let max_from_work = if naive_work > 0 {
        (200_000_000u64 / naive_work).max(5) as usize
    } else {
        base
    };

    base.min(max_from_work).max(5)
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

struct BenchResult {
    order: usize,
    scenario: String,
    image_size: String,
    radius: String,
    operator: String,
    input_pattern: String,
    naive_us: f64,
    prod_us: f64,
    speedup: f64,
    status: String,
    path: String,
}

fn bench_one(cfg: &Config, base_data: &[RGBA8]) -> BenchResult {
    let w = cfg.width;
    let h = cfg.height;
    let rx = cfg.rx;
    let ry = cfg.ry;
    let operator = cfg.operator;
    let iterations = cfg.iterations;

    let columns = std::cmp::min(rx.ceil() as u32 * 2, w);
    let rows = std::cmp::min(ry.ceil() as u32 * 2, h);
    let kernel_area = columns * rows;
    let uses_naive = kernel_area <= NAIVE_KERNEL_AREA_THRESHOLD;
    let path = if uses_naive { "naive" } else { "vhgw" };

    let op_name = match operator {
        MorphologyOperator::Erode => "Erode",
        MorphologyOperator::Dilate => "Dilate",
    };

    // --- Benchmark naive ---
    let mut naive_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut data = base_data.to_vec();
        let src = ImageRefMut::new(w, h, &mut data);
        let start = Instant::now();
        morphology::apply_naive_pub(operator, columns, rows, src);
        naive_times.push(start.elapsed());
        std::hint::black_box(&data);
    }

    // --- Benchmark production (apply) ---
    let mut prod_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut data = base_data.to_vec();
        let src = ImageRefMut::new(w, h, &mut data);
        let start = Instant::now();
        morphology::apply(operator, rx, ry, src);
        prod_times.push(start.elapsed());
        std::hint::black_box(&data);
    }

    let naive_median = median(&mut naive_times);
    let prod_median = median(&mut prod_times);

    let naive_us = naive_median.as_nanos() as f64 / 1000.0;
    let prod_us = prod_median.as_nanos() as f64 / 1000.0;
    let speedup = if prod_us > 0.0 {
        naive_us / prod_us
    } else {
        f64::INFINITY
    };

    let status = if speedup >= 1.05 {
        "FASTER".to_string()
    } else if speedup >= 0.95 {
        "SAME".to_string()
    } else {
        "REGRESSION".to_string()
    };

    BenchResult {
        order: cfg.order,
        scenario: cfg.scenario.to_string(),
        image_size: format!("{}x{}", w, h),
        radius: format!("rx={:.1},ry={:.1}", rx, ry),
        operator: op_name.to_string(),
        input_pattern: cfg.pattern.to_string(),
        naive_us,
        prod_us,
        speedup,
        status,
        path: path.to_string(),
    }
}

fn main() {
    let patterns: &[&'static str] = &["opaque", "text-like"];

    // -----------------------------------------------------------------------
    // Build scenario-based configurations
    // -----------------------------------------------------------------------
    let mut configs: Vec<Config> = Vec::new();
    let mut order = 0usize;

    // Helper: push configs for given scenario parameters across all patterns
    let add = |scenario: &'static str,
               width: u32,
               height: u32,
               rx: f32,
               ry: f32,
               operator: MorphologyOperator,
               configs: &mut Vec<Config>,
               order: &mut usize| {
        let iters = iterations_for_config(width, height, rx, ry);
        for &pattern in patterns {
            configs.push(Config {
                order: *order,
                scenario,
                width,
                height,
                rx,
                ry,
                operator,
                pattern,
                iterations: iters,
            });
            *order += 1;
        }
    };

    // ------------------------------------------------------------------
    // 1. Icon Text Outline: dilate r=2, icon sizes
    //    Most common real-world use. radius=2 -> 4x4=16 kernel -> vHGW path
    // ------------------------------------------------------------------
    for &(w, h) in &[(16, 16), (20, 20), (24, 24), (48, 48), (96, 96)] {
        add(
            "Icon Text Outline",
            w,
            h,
            2.0,
            2.0,
            MorphologyOperator::Dilate,
            &mut configs,
            &mut order,
        );
    }

    // ------------------------------------------------------------------
    // 2. Heading Outline: dilate r=3-4, medium/large text regions
    //    radius=3 -> 6x6=36 kernel, radius=4 -> 8x8=64 kernel -> vHGW
    // ------------------------------------------------------------------
    for &(w, h) in &[(200, 150), (400, 300), (600, 400)] {
        for &r in &[3.0, 4.0] {
            add(
                "Heading Outline",
                w,
                h,
                r,
                r,
                MorphologyOperator::Dilate,
                &mut configs,
                &mut order,
            );
        }
    }

    // ------------------------------------------------------------------
    // 3. Subtle Erode: erode r=1, various sizes
    //    radius=1 -> 2x2=4 kernel -> always naive path (area < 9)
    // ------------------------------------------------------------------
    for &(w, h) in &[
        (16, 16),
        (24, 24),
        (48, 48),
        (96, 96),
        (200, 150),
        (400, 300),
    ] {
        add(
            "Subtle Erode",
            w,
            h,
            1.0,
            1.0,
            MorphologyOperator::Erode,
            &mut configs,
            &mut order,
        );
    }

    // ------------------------------------------------------------------
    // 4. Thick Knockout: dilate r=8, medium/large -- stress test for vHGW
    //    radius=8 -> 16x16=256 kernel -> definitely vHGW
    // ------------------------------------------------------------------
    for &(w, h) in &[(96, 96), (200, 150), (400, 300)] {
        add(
            "Thick Knockout",
            w,
            h,
            8.0,
            8.0,
            MorphologyOperator::Dilate,
            &mut configs,
            &mut order,
        );
    }

    // ------------------------------------------------------------------
    // 5. Threshold Boundary: radii near NAIVE_KERNEL_AREA_THRESHOLD=9
    //    radius=1.0 -> 2x2=4 (naive), radius=1.5 -> 3x3=9 (boundary/naive),
    //    radius=2.0 -> 4x4=16 (vHGW)
    //    Test both operators at multiple sizes
    // ------------------------------------------------------------------
    for &(w, h) in &[(24, 24), (96, 96), (400, 300)] {
        for &r in &[1.0, 1.5, 2.0] {
            for &op in &[MorphologyOperator::Erode, MorphologyOperator::Dilate] {
                add(
                    "Threshold Boundary",
                    w,
                    h,
                    r,
                    r,
                    op,
                    &mut configs,
                    &mut order,
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // 6. Non-square Images: real aspect ratios with common dilate r=2
    //    Tests that vHGW handles non-square dimensions correctly
    // ------------------------------------------------------------------
    for &(w, h) in &[(200, 150), (400, 300), (600, 400)] {
        add(
            "Non-square",
            w,
            h,
            2.0,
            2.0,
            MorphologyOperator::Dilate,
            &mut configs,
            &mut order,
        );
    }

    // ------------------------------------------------------------------
    // 7. Asymmetric Radius: directional effects (rx!=ry)
    //    rx=1,ry=3 -> cols=2,rows=6 -> area=12 (vHGW)
    //    rx=3,ry=1 -> cols=6,rows=2 -> area=12 (vHGW)
    // ------------------------------------------------------------------
    for &(w, h) in &[(96, 96), (400, 300)] {
        for &(rx, ry) in &[(1.0, 3.0), (3.0, 1.0)] {
            add(
                "Asymmetric Radius",
                w,
                h,
                rx,
                ry,
                MorphologyOperator::Dilate,
                &mut configs,
                &mut order,
            );
        }
    }

    let total_configs = configs.len();
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    eprintln!(
        "Running {} configurations across 7 real-world scenarios",
        total_configs,
    );
    eprintln!("Using {} threads for parallel execution\n", num_threads);

    // -----------------------------------------------------------------------
    // Execute benchmarks in parallel using std::thread::scope
    // -----------------------------------------------------------------------
    let progress = AtomicUsize::new(0);

    let chunk_size = (configs.len() + num_threads - 1) / num_threads;
    let chunks: Vec<&[Config]> = configs.chunks(chunk_size).collect();

    let mut all_results: Vec<BenchResult> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let progress = &progress;
                s.spawn(move || {
                    let mut results = Vec::with_capacity(chunk.len());
                    for cfg in chunk {
                        let base_data = match cfg.pattern {
                            "opaque" => make_opaque(cfg.width, cfg.height, 42),
                            "text-like" => make_text_like(cfg.width, cfg.height, 99),
                            _ => unreachable!(),
                        };

                        let result = bench_one(cfg, &base_data);
                        results.push(result);

                        let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 20 == 0 || done == total_configs {
                            eprintln!("  Progress: {}/{}", done, total_configs);
                        }
                    }
                    results
                })
            })
            .collect();

        let mut merged = Vec::with_capacity(total_configs);
        for handle in handles {
            merged.extend(handle.join().unwrap());
        }
        merged
    });

    // Sort results back into original order
    all_results.sort_by_key(|r| r.order);

    // -----------------------------------------------------------------------
    // Print results table
    // -----------------------------------------------------------------------
    println!(
        "{:<20} | {:<12} | {:<16} | {:<8} | {:<10} | {:>6} | {:>10} | {:>10} | {:>8} | {:<10}",
        "Scenario",
        "Size",
        "Radius",
        "Operator",
        "Pattern",
        "Path",
        "Naive (us)",
        "Prod (us)",
        "Speedup",
        "Status"
    );
    println!("{}", "-".repeat(140));

    let mut prev_scenario = "";
    for result in &all_results {
        if result.scenario != prev_scenario {
            if !prev_scenario.is_empty() {
                println!("{}", "-".repeat(140));
            }
            prev_scenario = &result.scenario;
        }
        println!(
            "{:<20} | {:<12} | {:<16} | {:<8} | {:<10} | {:>6} | {:>10.2} | {:>10.2} | {:>7.2}x | {:<10}",
            result.scenario,
            result.image_size,
            result.radius,
            result.operator,
            result.input_pattern,
            result.path,
            result.naive_us,
            result.prod_us,
            result.speedup,
            result.status,
        );
    }

    // ---------------------------------------------------------------------------
    // Summary statistics
    // ---------------------------------------------------------------------------
    println!("\n{}", "=".repeat(140));
    println!("SUMMARY");
    println!("{}", "=".repeat(140));

    let regressions: Vec<&BenchResult> = all_results
        .iter()
        .filter(|r| r.status == "REGRESSION")
        .collect();
    println!("\nTotal configurations: {}", all_results.len());
    println!("Regressions (<0.95x): {}", regressions.len());
    println!(
        "Same (0.95-1.05x): {}",
        all_results.iter().filter(|r| r.status == "SAME").count()
    );
    println!(
        "Faster (>1.05x): {}",
        all_results.iter().filter(|r| r.status == "FASTER").count()
    );

    if !regressions.is_empty() {
        println!("\n--- REGRESSIONS ---");
        for r in &regressions {
            println!(
                "  [{}] {} {} {} {} | naive={:.2}us prod={:.2}us speedup={:.2}x (path={})",
                r.scenario,
                r.image_size,
                r.radius,
                r.operator,
                r.input_pattern,
                r.naive_us,
                r.prod_us,
                r.speedup,
                r.path
            );
        }
    }

    // Summary by scenario
    println!("\n--- BY SCENARIO ---");
    println!(
        "{:<20} | {:>6} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12}",
        "Scenario", "Count", "Faster", "Same", "Regress", "Avg Speedup", "Min Speedup"
    );
    let scenario_names = [
        "Icon Text Outline",
        "Heading Outline",
        "Subtle Erode",
        "Thick Knockout",
        "Threshold Boundary",
        "Non-square",
        "Asymmetric Radius",
    ];
    for scenario in &scenario_names {
        let subset: Vec<&BenchResult> = all_results
            .iter()
            .filter(|r| r.scenario == *scenario)
            .collect();
        if subset.is_empty() {
            continue;
        }
        let faster = subset.iter().filter(|r| r.status == "FASTER").count();
        let same = subset.iter().filter(|r| r.status == "SAME").count();
        let regress = subset.iter().filter(|r| r.status == "REGRESSION").count();
        let avg_speedup: f64 = subset.iter().map(|r| r.speedup).sum::<f64>() / subset.len() as f64;
        let min_speedup: f64 = subset
            .iter()
            .map(|r| r.speedup)
            .fold(f64::INFINITY, f64::min);
        println!(
            "{:<20} | {:>6} | {:>8} | {:>8} | {:>8} | {:>11.2}x | {:>11.2}x",
            scenario,
            subset.len(),
            faster,
            same,
            regress,
            avg_speedup,
            min_speedup
        );
    }

    // Summary by dispatch path
    println!("\n--- BY DISPATCH PATH ---");
    println!(
        "{:<8} | {:>6} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12}",
        "Path", "Count", "Faster", "Same", "Regress", "Avg Speedup", "Min Speedup"
    );
    for path_name in &["naive", "vhgw"] {
        let subset: Vec<&BenchResult> = all_results
            .iter()
            .filter(|r| r.path == *path_name)
            .collect();
        if subset.is_empty() {
            continue;
        }
        let faster = subset.iter().filter(|r| r.status == "FASTER").count();
        let same = subset.iter().filter(|r| r.status == "SAME").count();
        let regress = subset.iter().filter(|r| r.status == "REGRESSION").count();
        let avg_speedup: f64 = subset.iter().map(|r| r.speedup).sum::<f64>() / subset.len() as f64;
        let min_speedup: f64 = subset
            .iter()
            .map(|r| r.speedup)
            .fold(f64::INFINITY, f64::min);
        println!(
            "{:<8} | {:>6} | {:>8} | {:>8} | {:>8} | {:>11.2}x | {:>11.2}x",
            path_name,
            subset.len(),
            faster,
            same,
            regress,
            avg_speedup,
            min_speedup
        );
    }

    // Summary by operator
    println!("\n--- BY OPERATOR ---");
    println!(
        "{:<8} | {:>6} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12}",
        "Operator", "Count", "Faster", "Same", "Regress", "Avg Speedup", "Min Speedup"
    );
    for op_name in &["Erode", "Dilate"] {
        let subset: Vec<&BenchResult> = all_results
            .iter()
            .filter(|r| r.operator == *op_name)
            .collect();
        if subset.is_empty() {
            continue;
        }
        let faster = subset.iter().filter(|r| r.status == "FASTER").count();
        let same = subset.iter().filter(|r| r.status == "SAME").count();
        let regress = subset.iter().filter(|r| r.status == "REGRESSION").count();
        let avg_speedup: f64 = subset.iter().map(|r| r.speedup).sum::<f64>() / subset.len() as f64;
        let min_speedup: f64 = subset
            .iter()
            .map(|r| r.speedup)
            .fold(f64::INFINITY, f64::min);
        println!(
            "{:<8} | {:>6} | {:>8} | {:>8} | {:>8} | {:>11.2}x | {:>11.2}x",
            op_name,
            subset.len(),
            faster,
            same,
            regress,
            avg_speedup,
            min_speedup
        );
    }

    // Summary by input pattern
    println!("\n--- BY INPUT PATTERN ---");
    println!(
        "{:<10} | {:>6} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12}",
        "Pattern", "Count", "Faster", "Same", "Regress", "Avg Speedup", "Min Speedup"
    );
    for pat in patterns {
        let subset: Vec<&BenchResult> = all_results
            .iter()
            .filter(|r| r.input_pattern == *pat)
            .collect();
        if subset.is_empty() {
            continue;
        }
        let faster = subset.iter().filter(|r| r.status == "FASTER").count();
        let same = subset.iter().filter(|r| r.status == "SAME").count();
        let regress = subset.iter().filter(|r| r.status == "REGRESSION").count();
        let avg_speedup: f64 = subset.iter().map(|r| r.speedup).sum::<f64>() / subset.len() as f64;
        let min_speedup: f64 = subset
            .iter()
            .map(|r| r.speedup)
            .fold(f64::INFINITY, f64::min);
        println!(
            "{:<10} | {:>6} | {:>8} | {:>8} | {:>8} | {:>11.2}x | {:>11.2}x",
            pat,
            subset.len(),
            faster,
            same,
            regress,
            avg_speedup,
            min_speedup
        );
    }

    // Threshold analysis
    println!(
        "\n--- THRESHOLD ANALYSIS (NAIVE_KERNEL_AREA_THRESHOLD = {}) ---",
        NAIVE_KERNEL_AREA_THRESHOLD
    );
    println!("Production vs naive at the dispatch boundary (opaque pattern only):");
    let boundary_radii = [
        ("rx=1.0,ry=1.0", "2x2=4, well below threshold (naive)"),
        ("rx=1.5,ry=1.5", "3x3=9, at threshold boundary (naive)"),
        ("rx=2.0,ry=2.0", "4x4=16, above threshold (vHGW)"),
    ];
    for (radius_str, desc) in &boundary_radii {
        println!("\n  {} ({})", radius_str, desc);
        let subset: Vec<&BenchResult> = all_results
            .iter()
            .filter(|r| {
                r.radius == *radius_str
                    && r.scenario == "Threshold Boundary"
                    && r.input_pattern == "opaque"
            })
            .collect();
        for r in &subset {
            println!(
                "    {} {}: naive={:.2}us prod={:.2}us speedup={:.2}x [{}] ({})",
                r.image_size, r.operator, r.naive_us, r.prod_us, r.speedup, r.path, r.status
            );
        }
    }

    // Overall verdict
    println!("\n{}", "=".repeat(140));
    if regressions.is_empty() {
        println!("VERDICT: NO REGRESSIONS DETECTED");
    } else {
        println!(
            "VERDICT: {} REGRESSIONS DETECTED - review above",
            regressions.len()
        );
    }
    println!("{}", "=".repeat(140));
}
