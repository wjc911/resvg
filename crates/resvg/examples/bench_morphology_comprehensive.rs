// Comprehensive performance benchmark for feMorphology filter optimization.
//
// Tests naive O(n*r^2) vs production apply() (which dispatches to naive or vHGW)
// across all combinations of image sizes, radii, operators, and input patterns.
//
// Uses multithreading via std::thread::scope for faster execution.
//
// Run with:
//   cargo run --release --example bench_morphology_comprehensive

use resvg::filter::morphology;
use resvg::filter::ImageRefMut;
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
    size: u32,
    rx: f32,
    ry: f32,
    operator: MorphologyOperator,
    pattern: &'static str,
    iterations: usize,
}

// ---------------------------------------------------------------------------
// Image generators
// ---------------------------------------------------------------------------

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

fn make_gradient(w: u32, h: u32) -> Vec<RGBA8> {
    let mut data = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = ((x as f32 / w.max(1) as f32 + y as f32 / h.max(1) as f32) * 127.5) as u8;
            data.push(RGBA8 {
                r: v,
                g: v,
                b: v,
                a: 255,
            });
        }
    }
    data
}

fn make_sparse50(w: u32, h: u32, seed: u64) -> Vec<RGBA8> {
    let mut data = Vec::with_capacity((w * h) as usize);
    let mut state = seed;
    for _ in 0..(w * h) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bytes = state.to_le_bytes();
        let transparent = bytes[4] < 128;
        if transparent {
            data.push(RGBA8 {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            });
        } else {
            data.push(RGBA8 {
                r: bytes[0],
                g: bytes[1],
                b: bytes[2],
                a: bytes[3],
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

fn iterations_for_config(size: u32, rx: f32, ry: f32) -> usize {
    // Estimate work: pixels * kernel_area for naive path
    let pixels = (size as u64) * (size as u64);
    let columns = std::cmp::min(rx.ceil() as u64 * 2, size as u64);
    let rows = std::cmp::min(ry.ceil() as u64 * 2, size as u64);
    let kernel_area = columns * rows;
    let naive_work = pixels * kernel_area;

    // Target: each config completes within ~0.5 second for naive path.
    // Approximate cost: ~1ns per pixel*kernel element (rough estimate).
    // So 500_000_000 ns / (naive_work * 1ns) = iterations.
    let base = if size <= 32 {
        10_000
    } else if size <= 256 {
        1_000
    } else {
        100
    };

    // Cap iterations so naive doesn't run forever
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
    image_size: String,
    radius: String,
    operator: String,
    input_pattern: String,
    naive_us: f64,
    prod_us: f64,
    speedup: f64,
    status: String,
    path: String, // which path production used: "naive" or "vhgw"
}

fn bench_one(
    w: u32,
    h: u32,
    rx: f32,
    ry: f32,
    operator: MorphologyOperator,
    base_data: &[RGBA8],
    pattern_name: &str,
    iterations: usize,
) -> BenchResult {
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
        order: 0, // caller will set this
        image_size: format!("{}x{}", w, h),
        radius: format!("rx={:.1},ry={:.1}", rx, ry),
        operator: op_name.to_string(),
        input_pattern: pattern_name.to_string(),
        naive_us,
        prod_us,
        speedup,
        status,
        path: path.to_string(),
    }
}

fn main() {
    let sizes: &[u32] = &[4, 16, 32, 64, 128, 256, 512, 1024];

    let radii: &[(f32, f32, &str)] = &[
        (0.5, 0.5, "rx=0.5,ry=0.5 (1x1)"),
        (1.0, 1.0, "rx=1.0,ry=1.0 (2x2)"),
        (1.5, 1.5, "rx=1.5,ry=1.5 (3x3=9)"),
        (2.0, 2.0, "rx=2.0,ry=2.0 (4x4=16)"),
        (3.0, 3.0, "rx=3.0,ry=3.0 (6x6)"),
        (5.0, 5.0, "rx=5.0,ry=5.0 (10x10)"),
        (10.0, 10.0, "rx=10.0,ry=10.0 (20x20)"),
        (20.0, 20.0, "rx=20.0,ry=20.0 (40x40)"),
        (1.0, 5.0, "rx=1.0,ry=5.0 (asym)"),
        (5.0, 1.0, "rx=5.0,ry=1.0 (asym)"),
        (50.0, 50.0, "rx=50.0,ry=50.0 (extreme)"),
    ];

    let operators = [MorphologyOperator::Erode, MorphologyOperator::Dilate];
    let patterns: &[&'static str] = &["opaque", "gradient", "sparse50"];

    // -----------------------------------------------------------------------
    // Build all configurations upfront
    // -----------------------------------------------------------------------
    let mut configs: Vec<Config> = Vec::new();
    let mut order = 0usize;

    for &size in sizes {
        for &(rx, ry, _radius_label) in radii {
            let iters = iterations_for_config(size, rx, ry);
            for &op in &operators {
                for &pattern in patterns {
                    configs.push(Config {
                        order,
                        size,
                        rx,
                        ry,
                        operator: op,
                        pattern,
                        iterations: iters,
                    });
                    order += 1;
                }
            }
        }
    }

    let total_configs = configs.len();
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    eprintln!(
        "Running {} configurations ({} sizes x {} radii x {} operators x {} patterns)",
        total_configs,
        sizes.len(),
        radii.len(),
        operators.len(),
        patterns.len()
    );
    eprintln!(
        "Using {} threads for parallel execution\n",
        num_threads
    );

    // -----------------------------------------------------------------------
    // Execute benchmarks in parallel using std::thread::scope
    // -----------------------------------------------------------------------
    let progress = AtomicUsize::new(0);

    // Split configs into chunks, one per thread
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
                        let w = cfg.size;
                        let h = cfg.size;

                        let base_data = match cfg.pattern {
                            "opaque" => make_opaque(w, h, 42),
                            "gradient" => make_gradient(w, h),
                            "sparse50" => make_sparse50(w, h, 99),
                            _ => unreachable!(),
                        };

                        let mut result = bench_one(
                            w,
                            h,
                            cfg.rx,
                            cfg.ry,
                            cfg.operator,
                            &base_data,
                            cfg.pattern,
                            cfg.iterations,
                        );
                        result.order = cfg.order;
                        results.push(result);

                        let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 50 == 0 || done == total_configs {
                            eprintln!("  Progress: {}/{}", done, total_configs);
                        }
                    }
                    results
                })
            })
            .collect();

        // Collect results from all threads
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
        "{:<12} | {:<28} | {:<8} | {:<10} | {:>6} | {:>10} | {:>10} | {:>8} | {:<10}",
        "Image Size", "Radius", "Operator", "Pattern", "Path", "Naive (us)", "Prod (us)", "Speedup", "Status"
    );
    println!("{}", "-".repeat(120));

    for result in &all_results {
        println!(
            "{:<12} | {:<28} | {:<8} | {:<10} | {:>6} | {:>10.2} | {:>10.2} | {:>7.2}x | {:<10}",
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
    println!("\n{}", "=".repeat(120));
    println!("SUMMARY");
    println!("{}", "=".repeat(120));

    // Count regressions
    let regressions: Vec<&BenchResult> = all_results.iter().filter(|r| r.status == "REGRESSION").collect();
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
                "  {} {} {} {} | naive={:.2}us prod={:.2}us speedup={:.2}x (path={})",
                r.image_size, r.radius, r.operator, r.input_pattern, r.naive_us, r.prod_us, r.speedup, r.path
            );
        }
    }

    // Summary by image size
    println!("\n--- BY IMAGE SIZE ---");
    println!(
        "{:<12} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12}",
        "Size", "Faster", "Same", "Regress", "Avg Speedup", "Min Speedup"
    );
    for &size in sizes {
        let size_str = format!("{}x{}", size, size);
        let subset: Vec<&BenchResult> = all_results
            .iter()
            .filter(|r| r.image_size == size_str)
            .collect();
        let faster = subset.iter().filter(|r| r.status == "FASTER").count();
        let same = subset.iter().filter(|r| r.status == "SAME").count();
        let regress = subset.iter().filter(|r| r.status == "REGRESSION").count();
        let avg_speedup: f64 =
            subset.iter().map(|r| r.speedup).sum::<f64>() / subset.len() as f64;
        let min_speedup: f64 = subset
            .iter()
            .map(|r| r.speedup)
            .fold(f64::INFINITY, f64::min);
        println!(
            "{:<12} | {:>8} | {:>8} | {:>8} | {:>11.2}x | {:>11.2}x",
            size_str, faster, same, regress, avg_speedup, min_speedup
        );
    }

    // Summary by radius
    println!("\n--- BY RADIUS ---");
    println!(
        "{:<28} | {:>6} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12}",
        "Radius", "Path", "Faster", "Same", "Regress", "Avg Speedup", "Min Speedup"
    );
    for &(rx, ry, label) in radii {
        let radius_str = format!("rx={:.1},ry={:.1}", rx, ry);
        let subset: Vec<&BenchResult> = all_results
            .iter()
            .filter(|r| r.radius == radius_str)
            .collect();
        let path = if !subset.is_empty() {
            // For the path, pick any non-tiny image (e.g., 64x64) to show expected path
            subset
                .iter()
                .find(|r| r.image_size == "64x64")
                .map(|r| r.path.as_str())
                .unwrap_or(&subset[0].path)
        } else {
            "?"
        };
        let faster = subset.iter().filter(|r| r.status == "FASTER").count();
        let same = subset.iter().filter(|r| r.status == "SAME").count();
        let regress = subset.iter().filter(|r| r.status == "REGRESSION").count();
        let avg_speedup: f64 =
            subset.iter().map(|r| r.speedup).sum::<f64>() / subset.len().max(1) as f64;
        let min_speedup: f64 = subset
            .iter()
            .map(|r| r.speedup)
            .fold(f64::INFINITY, f64::min);
        println!(
            "{:<28} | {:>6} | {:>8} | {:>8} | {:>8} | {:>11.2}x | {:>11.2}x",
            label, path, faster, same, regress, avg_speedup, min_speedup
        );
    }

    // Summary by operator
    println!("\n--- BY OPERATOR ---");
    println!(
        "{:<8} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12}",
        "Operator", "Faster", "Same", "Regress", "Avg Speedup", "Min Speedup"
    );
    for op_name in &["Erode", "Dilate"] {
        let subset: Vec<&BenchResult> = all_results
            .iter()
            .filter(|r| r.operator == *op_name)
            .collect();
        let faster = subset.iter().filter(|r| r.status == "FASTER").count();
        let same = subset.iter().filter(|r| r.status == "SAME").count();
        let regress = subset.iter().filter(|r| r.status == "REGRESSION").count();
        let avg_speedup: f64 =
            subset.iter().map(|r| r.speedup).sum::<f64>() / subset.len().max(1) as f64;
        let min_speedup: f64 = subset
            .iter()
            .map(|r| r.speedup)
            .fold(f64::INFINITY, f64::min);
        println!(
            "{:<8} | {:>8} | {:>8} | {:>8} | {:>11.2}x | {:>11.2}x",
            op_name, faster, same, regress, avg_speedup, min_speedup
        );
    }

    // Summary by path (naive vs vhgw)
    println!("\n--- BY DISPATCH PATH ---");
    println!(
        "{:<8} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12}",
        "Path", "Faster", "Same", "Regress", "Avg Speedup", "Min Speedup"
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
        let avg_speedup: f64 =
            subset.iter().map(|r| r.speedup).sum::<f64>() / subset.len() as f64;
        let min_speedup: f64 = subset
            .iter()
            .map(|r| r.speedup)
            .fold(f64::INFINITY, f64::min);
        println!(
            "{:<8} | {:>8} | {:>8} | {:>8} | {:>11.2}x | {:>11.2}x",
            path_name, faster, same, regress, avg_speedup, min_speedup
        );
    }

    // Threshold analysis: look at the boundary cases
    println!("\n--- THRESHOLD ANALYSIS (NAIVE_KERNEL_AREA_THRESHOLD = 9) ---");
    println!("Comparing production (dispatched) vs always-naive at the boundary:");
    let boundary_radii = [
        ("rx=1.5,ry=1.5", "3x3=9, at threshold (naive)"),
        ("rx=2.0,ry=2.0", "4x4=16, just above (vhgw)"),
    ];
    for (radius_str, desc) in &boundary_radii {
        println!("\n  {} ({})", radius_str, desc);
        let subset: Vec<&BenchResult> = all_results
            .iter()
            .filter(|r| r.radius == *radius_str)
            .collect();
        for r in &subset {
            if r.input_pattern == "opaque" {
                println!(
                    "    {} {}: naive={:.2}us prod={:.2}us speedup={:.2}x ({})",
                    r.image_size, r.operator, r.naive_us, r.prod_us, r.speedup, r.status
                );
            }
        }
    }

    // Overall verdict
    println!("\n{}", "=".repeat(120));
    if regressions.is_empty() {
        println!("VERDICT: NO REGRESSIONS DETECTED");
    } else {
        println!(
            "VERDICT: {} REGRESSIONS DETECTED - review above",
            regressions.len()
        );
    }
    println!("{}", "=".repeat(120));
}
