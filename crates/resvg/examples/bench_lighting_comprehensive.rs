// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Comprehensive benchmark for feDiffuseLighting and feSpecularLighting filters.
//!
//! Tests multiple image sizes, light source types, parameter combinations,
//! and input patterns to detect performance regressions.
//!
//! Uses multithreading via `std::thread::scope` for faster execution across
//! available CPU cores.
//!
//! Usage: cargo run --release --example bench_lighting_comprehensive [diffuse|specular|both]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Input pattern types for benchmark diversity.
#[derive(Clone, Copy, Debug)]
enum InputPattern {
    /// All pixels have alpha=128 -- no surface variation, minimal normal computation.
    Flat,
    /// Linear alpha gradient from 0 to 255 -- typical use case.
    Gradient,
    /// Pseudo-random alpha values -- worst case for branch prediction / cache.
    Noisy,
}

impl InputPattern {
    fn name(&self) -> &'static str {
        match self {
            InputPattern::Flat => "flat",
            InputPattern::Gradient => "gradient",
            InputPattern::Noisy => "noisy",
        }
    }

    fn fill_svg_element(&self, w: u32, h: u32) -> String {
        match self {
            InputPattern::Flat => {
                // Solid gray rectangle -- uniform alpha
                format!(
                    r#"<rect width="{w}" height="{h}" fill="rgb(128,128,128)" fill-opacity="0.502"/>"#
                )
            }
            InputPattern::Gradient => {
                // Linear gradient from transparent to opaque
                format!(
                    r#"<defs>
    <linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="white" stop-opacity="0"/>
      <stop offset="1" stop-color="white" stop-opacity="1"/>
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#grad)"/>"#
                )
            }
            InputPattern::Noisy => {
                // Checkerboard-like pattern with varying alpha using nested rects
                // This creates surface variation through multiple overlapping elements.
                let mut elements =
                    format!(r#"<rect width="{w}" height="{h}" fill="gray" fill-opacity="0.3"/>"#);
                // Add some smaller rects at various positions to create alpha variation
                let step = (w.max(h) / 8).max(1);
                for i in 0..8 {
                    let x = (i * step) % w;
                    let y = (i * step) % h;
                    let sw = (step * 3).min(w - x);
                    let sh = (step * 2).min(h - y);
                    let opacity = 0.1 + (i as f32 * 0.1);
                    elements.push_str(&format!(
                        r#"<rect x="{x}" y="{y}" width="{sw}" height="{sh}" fill="white" fill-opacity="{opacity:.1}"/>"#
                    ));
                }
                elements
            }
        }
    }
}

/// Configuration for a single benchmark case.
struct BenchConfig {
    filter_type: &'static str,
    width: u32,
    height: u32,
    light_name: &'static str,
    light_xml: String,
    params_desc: String,
    filter_attrs: String,
    pattern: InputPattern,
}

fn generate_svg(config: &BenchConfig) -> String {
    let w = config.width;
    let h = config.height;
    let fill_elements = config.pattern.fill_svg_element(w, h);

    // Wrap fill elements in a group with the filter applied
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <filter id="f" x="0" y="0" width="100%" height="100%">
      <{filter_type} {attrs} lighting-color="white">
        {light}
      </{filter_type}>
    </filter>
  </defs>
  <g filter="url(#f)">
    {fill}
  </g>
</svg>"##,
        filter_type = config.filter_type,
        attrs = config.filter_attrs,
        light = config.light_xml,
        fill = fill_elements,
    )
}

fn pick_iterations(w: u32, h: u32) -> u32 {
    let pixels = w as u64 * h as u64;
    if pixels <= 16 {
        10000
    } else if pixels <= 256 {
        5000
    } else if pixels <= 1024 {
        2000
    } else if pixels <= 4096 {
        1000
    } else if pixels <= 16384 {
        200
    } else if pixels <= 65536 {
        100
    } else if pixels <= 262144 {
        30
    } else {
        10
    }
}

struct BenchResult {
    order: usize,
    filter_type: String,
    size: String,
    light_name: String,
    params: String,
    pattern: String,
    time_us: f64,
    mpix_per_s: f64,
    iterations: u32,
}

/// A self-contained configuration tuple for one benchmark run, including its
/// original index so results can be sorted back into order after parallel execution.
struct IndexedConfig {
    order: usize,
    config: BenchConfig,
}

fn run_bench(config: &BenchConfig, order: usize) -> BenchResult {
    let svg = generate_svg(config);
    let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
    let mut pixmap = tiny_skia::Pixmap::new(config.width, config.height).unwrap();

    // Warmup: 3 iterations
    for _ in 0..3 {
        resvg::render(
            &tree,
            tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
    }

    let iterations = pick_iterations(config.width, config.height);

    // Benchmark with multiple rounds for stability
    let mut best_time_us = f64::MAX;

    for _round in 0..3 {
        let start = Instant::now();
        for _ in 0..iterations {
            resvg::render(
                &tree,
                tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
        }
        let elapsed = start.elapsed();
        let per_iter_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
        if per_iter_us < best_time_us {
            best_time_us = per_iter_us;
        }
    }

    let pixels = config.width as f64 * config.height as f64;
    let mpix_per_s = pixels / best_time_us; // us -> s would multiply by 1e6, but Mpix divides by 1e6

    BenchResult {
        order,
        filter_type: config.filter_type.to_string(),
        size: format!("{}x{}", config.width, config.height),
        light_name: config.light_name.to_string(),
        params: config.params_desc.clone(),
        pattern: config.pattern.name().to_string(),
        time_us: best_time_us,
        mpix_per_s,
        iterations,
    }
}

/// Run benchmarks in parallel using `std::thread::scope`, splitting work across
/// available CPU cores.  An `AtomicUsize` progress counter lets every thread
/// report completions without a mutex.
fn run_benchmarks_parallel(configs: Vec<BenchConfig>) -> Vec<BenchResult> {
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let total = configs.len();
    eprintln!(
        "  Using {} threads for {} configurations",
        num_threads, total
    );

    // Wrap each config with its original index.
    let indexed: Vec<IndexedConfig> = configs
        .into_iter()
        .enumerate()
        .map(|(i, config)| IndexedConfig { order: i, config })
        .collect();

    // Split into roughly equal chunks, one per thread.
    let chunks: Vec<&[IndexedConfig]> = {
        let chunk_size = (indexed.len() + num_threads - 1) / num_threads;
        indexed.chunks(chunk_size).collect()
    };

    let progress = AtomicUsize::new(0);

    let mut all_results: Vec<BenchResult> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let progress_ref = &progress;
                s.spawn(move || {
                    let mut thread_results = Vec::with_capacity(chunk.len());
                    for item in chunk {
                        let result = run_bench(&item.config, item.order);
                        thread_results.push(result);

                        let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 10 == 0 || done == total {
                            eprint!(
                                "\r  Progress: {}/{} ({:.0}%)",
                                done,
                                total,
                                done as f64 / total as f64 * 100.0
                            );
                        }
                    }
                    thread_results
                })
            })
            .collect();

        // Collect results from all threads.
        let mut merged = Vec::with_capacity(total);
        for handle in handles {
            merged.extend(handle.join().unwrap());
        }
        merged
    });

    eprintln!();

    // Sort back into original configuration order.
    all_results.sort_by_key(|r| r.order);
    all_results
}

fn diffuse_configs() -> Vec<BenchConfig> {
    let sizes: Vec<(u32, u32)> = vec![
        (4, 4),
        (16, 16),
        (32, 32),
        (64, 64),
        (127, 127), // Just below threshold (128*128 = 16384, 127*127 = 16129)
        (128, 128), // At threshold
        (256, 256),
        (512, 512),
        (1024, 1024),
    ];

    let lights: Vec<(&str, String)> = vec![
        (
            "distant",
            r#"<feDistantLight azimuth="45" elevation="55"/>"#.to_string(),
        ),
        (
            "point",
            r#"<fePointLight x="150" y="60" z="200"/>"#.to_string(),
        ),
        (
            "spot",
            r#"<feSpotLight x="150" y="60" z="200" pointsAtX="100" pointsAtY="100" pointsAtZ="0" specularExponent="8" limitingConeAngle="30"/>"#.to_string(),
        ),
    ];

    let diffuse_params: Vec<(f32, f32, String)> = vec![
        (0.5, 1.0, "dc=0.5,ss=1".to_string()),
        (1.0, 1.0, "dc=1,ss=1".to_string()),
        (1.0, 5.0, "dc=1,ss=5".to_string()),
        (2.0, 5.0, "dc=2,ss=5".to_string()),
        (1.0, 10.0, "dc=1,ss=10".to_string()),
        (2.0, 10.0, "dc=2,ss=10".to_string()),
    ];

    let patterns = vec![
        InputPattern::Flat,
        InputPattern::Gradient,
        InputPattern::Noisy,
    ];

    let mut configs = Vec::new();

    for &(w, h) in &sizes {
        for &(light_name, ref light_xml) in &lights {
            for &(dc, ss, ref desc) in &diffuse_params {
                for &pattern in &patterns {
                    configs.push(BenchConfig {
                        filter_type: "feDiffuseLighting",
                        width: w,
                        height: h,
                        light_name,
                        light_xml: light_xml.clone(),
                        params_desc: desc.clone(),
                        filter_attrs: format!(r#"surfaceScale="{ss}" diffuseConstant="{dc}""#),
                        pattern,
                    });
                }
            }
        }
    }

    configs
}

fn specular_configs() -> Vec<BenchConfig> {
    let sizes: Vec<(u32, u32)> = vec![
        (4, 4),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ];

    let lights: Vec<(&str, String)> = vec![
        (
            "distant",
            r#"<feDistantLight azimuth="45" elevation="55"/>"#.to_string(),
        ),
        (
            "point",
            r#"<fePointLight x="150" y="60" z="200"/>"#.to_string(),
        ),
        (
            "spot",
            r#"<feSpotLight x="150" y="60" z="200" pointsAtX="100" pointsAtY="100" pointsAtZ="0" specularExponent="8" limitingConeAngle="30"/>"#.to_string(),
        ),
    ];

    let specular_params: Vec<(f32, f32, f32, String)> = vec![
        (1.0, 0.5, 1.0, "se=1,sc=0.5,ss=1".to_string()),
        (1.0, 1.0, 1.0, "se=1,sc=1,ss=1".to_string()),
        (5.0, 1.0, 1.0, "se=5,sc=1,ss=1".to_string()),
        (5.0, 0.5, 5.0, "se=5,sc=0.5,ss=5".to_string()),
        (20.0, 1.0, 1.0, "se=20,sc=1,ss=1".to_string()),
        (20.0, 1.0, 5.0, "se=20,sc=1,ss=5".to_string()),
        (128.0, 0.5, 1.0, "se=128,sc=0.5,ss=1".to_string()),
        (128.0, 1.0, 5.0, "se=128,sc=1,ss=5".to_string()),
    ];

    let patterns = vec![
        InputPattern::Flat,
        InputPattern::Gradient,
        InputPattern::Noisy,
    ];

    let mut configs = Vec::new();

    for &(w, h) in &sizes {
        for &(light_name, ref light_xml) in &lights {
            for &(se, sc, ss, ref desc) in &specular_params {
                for &pattern in &patterns {
                    configs.push(BenchConfig {
                        filter_type: "feSpecularLighting",
                        width: w,
                        height: h,
                        light_name,
                        light_xml: light_xml.clone(),
                        params_desc: desc.clone(),
                        filter_attrs: format!(
                            r#"surfaceScale="{ss}" specularConstant="{sc}" specularExponent="{se}""#
                        ),
                        pattern,
                    });
                }
            }
        }
    }

    configs
}

fn print_results_table(filter_name: &str, results: &[BenchResult]) {
    println!("\n{}", "=".repeat(120));
    println!("  {} Performance Results", filter_name);
    println!("{}", "=".repeat(120));
    println!(
        "{:<12} {:<10} {:<22} {:<10} {:>12} {:>12} {:>6}",
        "Size", "Light", "Params", "Input", "Time (us)", "Mpix/s", "Iters"
    );
    println!("{}", "-".repeat(120));

    let mut prev_size = String::new();
    for r in results {
        // Add separator between size groups
        if r.size != prev_size && !prev_size.is_empty() {
            println!("{}", "-".repeat(120));
        }
        prev_size = r.size.clone();

        println!(
            "{:<12} {:<10} {:<22} {:<10} {:>12.1} {:>12.2} {:>6}",
            r.size, r.light_name, r.params, r.pattern, r.time_us, r.mpix_per_s, r.iterations
        );
    }
    println!("{}", "=".repeat(120));
}

/// Print a summary table grouped by size and light type, averaging across params/patterns.
fn print_summary_table(filter_name: &str, results: &[BenchResult]) {
    println!("\n{}", "=".repeat(90));
    println!(
        "  {} Summary (averaged across params and input patterns)",
        filter_name
    );
    println!("{}", "=".repeat(90));
    println!(
        "{:<12} {:<10} {:>15} {:>15} {:>12} {:>8}",
        "Size", "Light", "Avg Time (us)", "Med Time (us)", "Avg Mpix/s", "Samples"
    );
    println!("{}", "-".repeat(90));

    // Group by (size, light_name)
    let mut groups: std::collections::BTreeMap<(String, String), Vec<f64>> =
        std::collections::BTreeMap::new();

    for r in results {
        groups
            .entry((r.size.clone(), r.light_name.clone()))
            .or_default()
            .push(r.time_us);
    }

    let mut prev_size = String::new();
    for ((size, light), mut times) in groups {
        if size != prev_size && !prev_size.is_empty() {
            println!("{}", "-".repeat(90));
        }
        prev_size = size.clone();

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let median = if times.len() % 2 == 0 {
            (times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.0
        } else {
            times[times.len() / 2]
        };

        // Parse size for mpix calculation
        let parts: Vec<&str> = size.split('x').collect();
        let w: f64 = parts[0].parse().unwrap();
        let h: f64 = parts[1].parse().unwrap();
        let mpix = w * h / avg;

        println!(
            "{:<12} {:<10} {:>15.1} {:>15.1} {:>12.2} {:>8}",
            size,
            light,
            avg,
            median,
            mpix,
            times.len()
        );
    }
    println!("{}", "=".repeat(90));
}

/// Run a focused threshold test for feDiffuseLighting around the 128x128 crossover.
fn run_threshold_test() {
    println!("\n{}", "=".repeat(100));
    println!("  feDiffuseLighting Threshold Test (naive <128x128, optimized >=128x128)");
    println!("{}", "=".repeat(100));

    let threshold_sizes: Vec<(u32, u32, &str)> = vec![
        (64, 64, "well-below"),
        (100, 100, "below"),
        (120, 120, "near-below"),
        (127, 127, "just-below (naive)"),
        (128, 128, "at-threshold (optimized)"),
        (129, 129, "just-above (optimized)"),
        (140, 140, "above"),
        (160, 160, "well-above"),
        (256, 256, "far-above"),
    ];

    let light_xml = r#"<feDistantLight azimuth="45" elevation="55"/>"#;

    println!(
        "{:<24} {:<18} {:>12} {:>12} {:>10}",
        "Size", "Category", "Time (us)", "Mpix/s", "Path"
    );
    println!("{}", "-".repeat(100));

    // Build configs for parallel execution of threshold test.
    let configs: Vec<BenchConfig> = threshold_sizes
        .iter()
        .map(|&(w, h, _)| BenchConfig {
            filter_type: "feDiffuseLighting",
            width: w,
            height: h,
            light_name: "distant",
            light_xml: light_xml.to_string(),
            params_desc: "dc=1,ss=5".to_string(),
            filter_attrs: r#"surfaceScale="5" diffuseConstant="1""#.to_string(),
            pattern: InputPattern::Gradient,
        })
        .collect();

    let results = run_benchmarks_parallel(configs);

    for (result, &(w, h, category)) in results.iter().zip(threshold_sizes.iter()) {
        let path = if (w * h) < 128 * 128 {
            "NAIVE"
        } else {
            "OPTIMIZED"
        };

        println!(
            "{:<24} {:<18} {:>12.1} {:>12.2} {:>10}",
            result.size, category, result.time_us, result.mpix_per_s, path
        );
    }
    println!("{}", "=".repeat(100));
}

/// Check for potential regressions: flag any case where small images are anomalously slow.
fn check_regressions(results: &[BenchResult], filter_name: &str) {
    println!("\n{}", "=".repeat(80));
    println!(
        "  {} Regression Check (>5% slower than expected)",
        filter_name
    );
    println!("{}", "=".repeat(80));

    // Group results by (light, params, pattern) and check that larger images
    // have proportionally better or equal Mpix/s throughput.
    // A regression would be if Mpix/s drops significantly at a given size.

    let mut groups: std::collections::HashMap<String, Vec<(u32, u32, f64, f64)>> =
        std::collections::HashMap::new();

    for r in results {
        if r.filter_type != filter_name {
            continue;
        }
        let key = format!("{}_{}_{}", r.light_name, r.params, r.pattern);
        let parts: Vec<&str> = r.size.split('x').collect();
        let w: u32 = parts[0].parse().unwrap();
        let h: u32 = parts[1].parse().unwrap();
        groups
            .entry(key)
            .or_default()
            .push((w, h, r.time_us, r.mpix_per_s));
    }

    let mut regressions_found = false;

    for (key, mut entries) in groups {
        entries.sort_by_key(|&(w, h, _, _)| w * h);

        // Check that throughput (Mpix/s) generally increases or stays stable
        // as image size grows (larger images amortize overhead better).
        // Flag cases where Mpix/s drops by >5% compared to the previous size.
        for i in 1..entries.len() {
            let (pw, ph, _pt, prev_mpix) = entries[i - 1];
            let (cw, ch, _ct, cur_mpix) = entries[i];

            // Only flag if current throughput is significantly lower than previous
            // AND the current size is large enough to be meaningful (>= 32x32)
            if cw * ch >= 32 * 32 && prev_mpix > 0.0 {
                let ratio = cur_mpix / prev_mpix;
                if ratio < 0.95 {
                    println!(
                        "  WARNING: {} -- {}x{} ({:.2} Mpix/s) is {:.1}% slower than {}x{} ({:.2} Mpix/s)",
                        key,
                        cw,
                        ch,
                        cur_mpix,
                        (1.0 - ratio) * 100.0,
                        pw,
                        ph,
                        prev_mpix,
                    );
                    regressions_found = true;
                }
            }
        }
    }

    if !regressions_found {
        println!("  No regressions detected (all throughput changes within 5% tolerance).");
    }
    println!("{}", "=".repeat(80));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("both");

    let run_diffuse = mode == "diffuse" || mode == "both";
    let run_specular = mode == "specular" || mode == "both";

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("Comprehensive Lighting Filter Benchmark");
    println!("Mode: {}", mode);
    println!("Threads: {}", num_threads);
    println!("Timestamp: {:?}", std::time::SystemTime::now());
    println!();

    if run_diffuse {
        println!("Generating feDiffuseLighting benchmark configurations...");
        let configs = diffuse_configs();
        println!("Running {} feDiffuseLighting benchmarks...", configs.len());

        let results = run_benchmarks_parallel(configs);

        print_results_table("feDiffuseLighting", &results);
        print_summary_table("feDiffuseLighting", &results);
        run_threshold_test();
        check_regressions(&results, "feDiffuseLighting");
    }

    if run_specular {
        println!("\nGenerating feSpecularLighting benchmark configurations...");
        let configs = specular_configs();
        println!("Running {} feSpecularLighting benchmarks...", configs.len());

        let results = run_benchmarks_parallel(configs);

        print_results_table("feSpecularLighting", &results);
        print_summary_table("feSpecularLighting", &results);
        check_regressions(&results, "feSpecularLighting");
    }

    println!("\nBenchmark complete.");
}
