// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! End-to-end benchmark for SVG filter optimizations.
//!
//! Renders programmatically-generated SVGs (realistic filter scenarios) and
//! existing filter test SVGs at multiple resolutions, measuring wall-clock time.
//! Supports sequential (--threads 1) or parallel (--threads N) execution.
//!
//! **Modes:**
//! - Default: run benchmark, output TSV to stdout
//! - `--compare <baseline.tsv>`: run benchmark and compare against a baseline file
//! - `--output-tsv <file>`: save TSV output to file instead of stdout
//! - `--threads N`: use N threads (default: 1 for sequential, noise-free measurement)
//!
//! Usage:
//!   cargo run --release --example bench_e2e -- --output-tsv /tmp/baseline.tsv
//!   cargo run --release --example bench_e2e -- --compare /tmp/baseline.tsv

use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration (defaults, overridable via CLI)
// ---------------------------------------------------------------------------

/// Representative resolutions (width, height).
const RESOLUTIONS: &[(u32, u32)] = &[
    (16, 16),
    (20, 20),
    (24, 24),
    (48, 48),
    (96, 96),
    (200, 150),
    (400, 300),
    (600, 400),
    (800, 600),
    (1024, 768),
    (1500, 1000),
];

/// Resolution-scaled iteration count. Larger images require fewer iterations
/// to amortise warmup cost while still suppressing sub-percent noise.
fn iters_for_resolution(w: u32, h: u32) -> usize {
    let max_dim = w.max(h);
    match max_dim {
        0..=49 => 2000,
        50..=99 => 1000,
        100..=499 => 300,
        _ => 100,
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

struct TestCase {
    name: String,
    width: u32,
    height: u32,
    svg: String,
}

struct BenchResult {
    name: String,
    resolution: String,
    median_ms: f64,
}

// ---------------------------------------------------------------------------
// SVG generation helpers
// ---------------------------------------------------------------------------

fn svg_wrap(width: u32, height: u32, filter_defs: &str, body: &str) -> String {
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>{filter_defs}</defs>
  {body}
</svg>"##,
    )
}

fn rect_with_filter(width: u32, height: u32, filter_id: &str) -> String {
    format!(
        r#"<rect x="0" y="0" width="{width}" height="{height}" fill="seagreen" filter="url(#{filter_id})"/>"#,
    )
}

fn two_rects_with_filter(width: u32, height: u32, filter_id: &str) -> String {
    format!(
        r#"<rect x="0" y="0" width="{width}" height="{height}" fill="seagreen" filter="url(#{filter_id})"/>
<rect x="{}" y="{}" width="{}" height="{}" fill="coral" filter="url(#{filter_id})"/>"#,
        width / 4,
        height / 4,
        width / 2,
        height / 2,
    )
}

// ---------------------------------------------------------------------------
// Test case generators: Single-filter scenarios
// ---------------------------------------------------------------------------

fn gen_drop_shadow(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feGaussianBlur in="SourceAlpha" stdDeviation="4" result="blur"/>
      <feOffset in="blur" dx="1" dy="1" result="offset"/>
      <feComposite in="SourceGraphic" in2="offset" operator="over"/>
    </filter>"#;
    TestCase {
        name: "drop-shadow".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_soft_blur(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feGaussianBlur stdDeviation="4"/>
    </filter>"#;
    TestCase {
        name: "soft-blur".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_backdrop_blur(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feGaussianBlur stdDeviation="16"/>
    </filter>"#;
    TestCase {
        name: "backdrop-blur".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_color_desaturate(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feColorMatrix type="saturate" values="0.3"/>
    </filter>"#;
    TestCase {
        name: "color-desaturate".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_hue_rotate(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feColorMatrix type="hueRotate" values="90"/>
    </filter>"#;
    TestCase {
        name: "hue-rotate".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_gamma_correct(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feComponentTransfer>
        <feFuncR type="gamma" amplitude="1" exponent="0.45" offset="0"/>
        <feFuncG type="gamma" amplitude="1" exponent="0.45" offset="0"/>
        <feFuncB type="gamma" amplitude="1" exponent="0.45" offset="0"/>
      </feComponentTransfer>
    </filter>"#;
    TestCase {
        name: "gamma-correct".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_sharpen(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feConvolveMatrix order="3" kernelMatrix="0 -1 0 -1 5 -1 0 -1 0"/>
    </filter>"#;
    TestCase {
        name: "sharpen".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_noise_fill(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feTurbulence type="fractalNoise" baseFrequency="0.05" numOctaves="2"/>
    </filter>"#;
    TestCase {
        name: "noise-fill".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_text_outline(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feMorphology operator="dilate" radius="2" in="SourceGraphic" result="dilated"/>
      <feComposite in="dilated" in2="SourceGraphic" operator="out"/>
    </filter>"#;
    TestCase {
        name: "text-outline".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_erode(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feMorphology operator="erode" radius="1"/>
    </filter>"#;
    TestCase {
        name: "erode".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_arithmetic_blend(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feFlood flood-color="coral" flood-opacity="0.5" result="flood"/>
      <feComposite in="SourceGraphic" in2="flood" operator="arithmetic" k1="0" k2="0.5" k3="0.5" k4="0"/>
    </filter>"#;
    TestCase {
        name: "arithmetic-blend".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

// ---------------------------------------------------------------------------
// Test case generators: Combination filter scenarios
// ---------------------------------------------------------------------------

fn gen_3d_button(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feGaussianBlur in="SourceAlpha" stdDeviation="2" result="blur"/>
      <feDiffuseLighting in="blur" surfaceScale="5" diffuseConstant="0.75" lighting-color="white" result="diffuse">
        <feDistantLight azimuth="45" elevation="55"/>
      </feDiffuseLighting>
      <feSpecularLighting in="blur" surfaceScale="5" specularConstant="0.5" specularExponent="10" lighting-color="white" result="specular">
        <feDistantLight azimuth="45" elevation="55"/>
      </feSpecularLighting>
      <feComposite in="diffuse" in2="SourceGraphic" operator="in" result="lit"/>
      <feComposite in="specular" in2="lit" operator="over"/>
    </filter>"#;
    TestCase {
        name: "3d-button".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_icon_glow(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur"/>
      <feComposite in="blur" in2="SourceGraphic" operator="arithmetic" k1="0" k2="1" k3="0.8" k4="0" result="glow"/>
      <feMerge>
        <feMergeNode in="glow"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>"#;
    TestCase {
        name: "icon-glow".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_inner_shadow(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feGaussianBlur in="SourceAlpha" stdDeviation="2" result="blur"/>
      <feOffset in="blur" dx="2" dy="2" result="offset"/>
      <feComposite in="offset" in2="SourceGraphic" operator="in" result="shadow"/>
      <feMerge>
        <feMergeNode in="SourceGraphic"/>
        <feMergeNode in="shadow"/>
      </feMerge>
    </filter>"#;
    TestCase {
        name: "inner-shadow".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

fn gen_card_ui(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feColorMatrix type="saturate" values="1.2" result="saturated"/>
      <feGaussianBlur in="SourceAlpha" stdDeviation="3" result="shadow-blur"/>
      <feOffset in="shadow-blur" dx="2" dy="2" result="shadow"/>
      <feMerge>
        <feMergeNode in="shadow"/>
        <feMergeNode in="saturated"/>
      </feMerge>
    </filter>"#;
    TestCase {
        name: "card-ui".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &two_rects_with_filter(w, h, "f")),
    }
}

fn gen_emboss_text(w: u32, h: u32) -> TestCase {
    let filter = r#"
    <filter id="f">
      <feConvolveMatrix order="3" kernelMatrix="-2 -1 0 -1 1 1 0 1 2" divisor="1" result="emboss"/>
      <feComposite in="emboss" in2="SourceGraphic" operator="in"/>
    </filter>"#;
    TestCase {
        name: "emboss-text".into(),
        width: w,
        height: h,
        svg: svg_wrap(w, h, filter, &rect_with_filter(w, h, "f")),
    }
}

// ---------------------------------------------------------------------------
// Resolution range for each scenario
// ---------------------------------------------------------------------------

/// Returns the indices into RESOLUTIONS that are applicable for this scenario.
fn applicable_resolutions(name: &str) -> Vec<usize> {
    let res = RESOLUTIONS;
    match name {
        // 16-1500 (all sizes - most common filter)
        "drop-shadow" => (0..res.len()).collect(),
        // 200-1500 (small blur is pointless)
        "soft-blur" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w >= 200)
            .map(|(i, _)| i)
            .collect(),
        // 200-1500 (iOS frosted glass)
        "backdrop-blur" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w >= 200)
            .map(|(i, _)| i)
            .collect(),
        // 16-1500 (all sizes - icons hover gray)
        "color-desaturate" => (0..res.len()).collect(),
        // 16-1024 (icons to laptop)
        "hue-rotate" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w <= 1024)
            .map(|(i, _)| i)
            .collect(),
        // 200-1024 (applied to images not icons)
        "gamma-correct" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w >= 200 && *w <= 1024)
            .map(|(i, _)| i)
            .collect(),
        // 400-1500 (sharpen on large images)
        "sharpen" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w >= 400)
            .map(|(i, _)| i)
            .collect(),
        // 200-1024 (texture generation)
        "noise-fill" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w >= 200 && *w <= 1024)
            .map(|(i, _)| i)
            .collect(),
        // 16-400 (text/icon elements)
        "text-outline" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w <= 400)
            .map(|(i, _)| i)
            .collect(),
        // 16-600 (small to medium elements)
        "erode" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w <= 600)
            .map(|(i, _)| i)
            .collect(),
        // 200-1024 (medium range)
        "arithmetic-blend" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w >= 200 && *w <= 1024)
            .map(|(i, _)| i)
            .collect(),
        // 24-400 (small buttons to medium)
        "3d-button" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w >= 24 && *w <= 400)
            .map(|(i, _)| i)
            .collect(),
        // 16-200 (icons only)
        "icon-glow" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w <= 200)
            .map(|(i, _)| i)
            .collect(),
        // 24-600 (small to medium UI elements)
        "inner-shadow" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w >= 24 && *w <= 600)
            .map(|(i, _)| i)
            .collect(),
        // 200-800 (card components)
        "card-ui" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w >= 200 && *w <= 800)
            .map(|(i, _)| i)
            .collect(),
        // 200-600 (text regions)
        "emboss-text" => res
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w >= 200 && *w <= 600)
            .map(|(i, _)| i)
            .collect(),
        _ => (0..res.len()).collect(),
    }
}

// ---------------------------------------------------------------------------
// Generate all programmatic test cases
// ---------------------------------------------------------------------------

fn generate_all_cases() -> Vec<TestCase> {
    let generators: Vec<(&str, fn(u32, u32) -> TestCase)> = vec![
        ("drop-shadow", gen_drop_shadow),
        ("soft-blur", gen_soft_blur),
        ("backdrop-blur", gen_backdrop_blur),
        ("color-desaturate", gen_color_desaturate),
        ("hue-rotate", gen_hue_rotate),
        ("gamma-correct", gen_gamma_correct),
        ("sharpen", gen_sharpen),
        ("noise-fill", gen_noise_fill),
        ("text-outline", gen_text_outline),
        ("erode", gen_erode),
        ("arithmetic-blend", gen_arithmetic_blend),
        ("3d-button", gen_3d_button),
        ("icon-glow", gen_icon_glow),
        ("inner-shadow", gen_inner_shadow),
        ("card-ui", gen_card_ui),
        ("emboss-text", gen_emboss_text),
    ];

    let mut cases = Vec::new();
    for (name, generator) in &generators {
        for idx in applicable_resolutions(name) {
            let (w, h) = RESOLUTIONS[idx];
            cases.push(generator(w, h));
        }
    }
    cases
}

// ---------------------------------------------------------------------------
// Collect existing filter test SVGs
// ---------------------------------------------------------------------------

fn find_filter_test_dir() -> Option<PathBuf> {
    // Try relative to the executable location first, then standard paths.
    let candidates = [
        PathBuf::from("crates/resvg/tests/tests/filters"),
        PathBuf::from("tests/tests/filters"),
    ];
    // Also try from the manifest directory (running via cargo run)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").ok();
    let mut all = candidates.to_vec();
    if let Some(dir) = &manifest_dir {
        all.insert(0, PathBuf::from(dir).join("tests/tests/filters"));
    }
    all.into_iter().find(|p| p.is_dir())
}

fn collect_existing_filter_svgs() -> Vec<TestCase> {
    let filter_dir = match find_filter_test_dir() {
        Some(d) => d,
        None => {
            eprintln!("[warn] Filter test directory not found, skipping existing SVGs");
            return Vec::new();
        }
    };

    let mut cases = Vec::new();
    let mut svg_paths: Vec<PathBuf> = Vec::new();

    // Walk directories
    if let Ok(entries) = std::fs::read_dir(&filter_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Ok(files) = std::fs::read_dir(&path) {
                    for file in files.flatten() {
                        let fpath = file.path();
                        if fpath.extension().is_some_and(|e| e == "svg") {
                            svg_paths.push(fpath);
                        }
                    }
                }
            }
        }
    }

    svg_paths.sort();

    for svg_path in &svg_paths {
        let svg_data = match std::fs::read_to_string(svg_path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Derive a name from path: "feGaussianBlur/simple-case"
        let rel = svg_path
            .strip_prefix(&filter_dir)
            .unwrap_or(svg_path)
            .with_extension("");
        let name = format!("existing:{}", rel.display());

        // Parse to get original dimensions
        let opt = usvg::Options::default();
        let tree = match usvg::Tree::from_str(&svg_data, &opt) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let orig_size = tree.size().to_int_size();
        let ow = orig_size.width().max(1);
        let oh = orig_size.height().max(1);

        // 1x (original size)
        cases.push(TestCase {
            name: format!("{name}@1x"),
            width: ow,
            height: oh,
            svg: svg_data.clone(),
        });

        // 3x scale
        cases.push(TestCase {
            name: format!("{name}@3x"),
            width: ow * 3,
            height: oh * 3,
            svg: svg_data,
        });
    }

    cases
}

// Maximum wall time budget per case (milliseconds). Cases that are intrinsically
// slower than this per-iteration (e.g. huge-radius morphology) are auto-scaled.
const MAX_CASE_BUDGET_MS: f64 = 30_000.0; // 30 seconds total per case
const SKIP_ITER_THRESHOLD_MS: f64 = 10_000.0; // if 1 iter > 10s, skip case

// ---------------------------------------------------------------------------
// Benchmark execution
// ---------------------------------------------------------------------------

fn bench_one(case: &TestCase) -> f64 {
    let opt = usvg::Options::default();
    let tree = match usvg::Tree::from_str(&case.svg, &opt) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[warn] Failed to parse SVG for '{}': {}", case.name, e);
            return -1.0;
        }
    };

    let orig_size = tree.size().to_int_size();
    let sx = case.width as f32 / orig_size.width().max(1) as f32;
    let sy = case.height as f32 / orig_size.height().max(1) as f32;
    let transform = tiny_skia::Transform::from_scale(sx, sy);

    let mut pixmap = match tiny_skia::Pixmap::new(case.width, case.height) {
        Some(p) => p,
        None => {
            eprintln!(
                "[warn] Failed to create pixmap {}x{} for '{}'",
                case.width, case.height, case.name
            );
            return -1.0;
        }
    };

    // Probe: run one iteration to estimate per-iter cost and scale accordingly.
    pixmap.fill(tiny_skia::Color::TRANSPARENT);
    let probe_start = Instant::now();
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    let probe_ms = probe_start.elapsed().as_secs_f64() * 1000.0;

    if probe_ms >= SKIP_ITER_THRESHOLD_MS {
        // Degenerate case (e.g. huge-radius morphology): skip with probe as result.
        eprintln!(" [slow:{:.0}ms, skipping extra iters]", probe_ms);
        return probe_ms;
    }

    // Scale iteration count to stay within budget, but respect resolution-based minimum.
    let resolution_iters = iters_for_resolution(case.width, case.height);
    let budget_iters = if probe_ms > 0.0 {
        (MAX_CASE_BUDGET_MS / probe_ms) as usize
    } else {
        resolution_iters
    };
    let bench_iters = resolution_iters.min(budget_iters).max(5);
    let warmup_iters = (bench_iters / 10).max(2);

    // Warmup (probe already counted as 1)
    for _ in 1..warmup_iters {
        pixmap.fill(tiny_skia::Color::TRANSPARENT);
        resvg::render(&tree, transform, &mut pixmap.as_mut());
    }

    // Measure
    let mut times = Vec::with_capacity(bench_iters);
    for _ in 0..bench_iters {
        pixmap.fill(tiny_skia::Color::TRANSPARENT);
        let start = Instant::now();
        resvg::render(&tree, transform, &mut pixmap.as_mut());
        times.push(start.elapsed().as_secs_f64() * 1000.0); // ms
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2] // median
}

fn run_bench(cases: Vec<TestCase>, num_threads: usize) -> Vec<BenchResult> {
    let total = cases.len();
    let counter = AtomicUsize::new(0);
    let cases_ref = &cases;
    let counter_ref = &counter;

    // Pre-compute the work assignments for each thread
    let chunks: Vec<Vec<usize>> = {
        let n = num_threads.max(1);
        let mut chunks = vec![Vec::new(); n];
        for (i, _) in cases.iter().enumerate() {
            chunks[i % n].push(i);
        }
        chunks
    };

    let all_results: Vec<Vec<(usize, BenchResult)>> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|indices| {
                s.spawn(move || {
                    let mut results = Vec::new();
                    for idx in indices {
                        let case = &cases_ref[idx];
                        let median = bench_one(case);
                        let done = counter_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        eprint!("\r[{}/{}] {}", done, total, case.name);
                        results.push((
                            idx,
                            BenchResult {
                                name: case.name.clone(),
                                resolution: format!("{}x{}", case.width, case.height),
                                median_ms: median,
                            },
                        ));
                    }
                    results
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    eprintln!();

    // Flatten and sort by original index
    let mut flat: Vec<(usize, BenchResult)> = all_results.into_iter().flatten().collect();
    flat.sort_by_key(|(idx, _)| *idx);
    flat.into_iter().map(|(_, r)| r).collect()
}

// ---------------------------------------------------------------------------
// Output & comparison
// ---------------------------------------------------------------------------

fn write_tsv(results: &[BenchResult], writer: &mut dyn std::io::Write) {
    writeln!(writer, "name\tresolution\tmedian_ms").unwrap();
    for r in results {
        writeln!(writer, "{}\t{}\t{:.3}", r.name, r.resolution, r.median_ms).unwrap();
    }
}

fn load_tsv(path: &Path) -> Vec<BenchResult> {
    let file = std::fs::File::open(path).expect("Failed to open baseline TSV");
    let reader = std::io::BufReader::new(file);
    let mut results = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with("name\t") {
            continue; // skip header
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 3 {
            results.push(BenchResult {
                name: parts[0].to_string(),
                resolution: parts[1].to_string(),
                median_ms: parts[2].parse().unwrap_or(-1.0),
            });
        }
    }
    results
}

fn compare_results(baseline: &[BenchResult], optimized: &[BenchResult]) {
    // Build a lookup: (name, resolution) -> median_ms
    let baseline_map: HashMap<(&str, &str), f64> = baseline
        .iter()
        .map(|r| ((r.name.as_str(), r.resolution.as_str()), r.median_ms))
        .collect();

    let _optimized_map: HashMap<(&str, &str), f64> = optimized
        .iter()
        .map(|r| ((r.name.as_str(), r.resolution.as_str()), r.median_ms))
        .collect();

    // Separate generated vs existing test cases
    let mut gen_pairs: Vec<(&str, &str, f64, f64)> = Vec::new();
    let mut exist_1x_base = 0.0_f64;
    let mut exist_1x_opt = 0.0_f64;
    let mut exist_3x_base = 0.0_f64;
    let mut exist_3x_opt = 0.0_f64;
    let mut exist_1x_count = 0usize;
    let mut exist_3x_count = 0usize;

    // Per-filter type stats
    struct FilterStats {
        count: usize,
        speedups: Vec<f64>,
    }
    let mut filter_stats: HashMap<String, FilterStats> = HashMap::new();

    let mut regressions: Vec<(String, String, f64, f64)> = Vec::new();

    for r in optimized {
        let key = (r.name.as_str(), r.resolution.as_str());
        if let Some(&base_ms) = baseline_map.get(&key) {
            if base_ms <= 0.0 || r.median_ms <= 0.0 {
                continue;
            }
            let speedup = base_ms / r.median_ms;

            if r.name.starts_with("existing:") {
                // Extract filter type from path like "existing:feGaussianBlur/simple-case@1x"
                let inner = r.name.strip_prefix("existing:").unwrap_or(&r.name);
                let filter_type = inner.split('/').next().unwrap_or("unknown");
                let entry = filter_stats
                    .entry(filter_type.to_string())
                    .or_insert_with(|| FilterStats {
                        count: 0,
                        speedups: Vec::new(),
                    });
                entry.count += 1;
                entry.speedups.push(speedup);

                if r.name.ends_with("@1x") {
                    exist_1x_base += base_ms;
                    exist_1x_opt += r.median_ms;
                    exist_1x_count += 1;
                } else if r.name.ends_with("@3x") {
                    exist_3x_base += base_ms;
                    exist_3x_opt += r.median_ms;
                    exist_3x_count += 1;
                }
            } else {
                gen_pairs.push((&r.name, &r.resolution, base_ms, r.median_ms));
            }

            if speedup < 0.95 {
                regressions.push((r.name.clone(), r.resolution.clone(), base_ms, r.median_ms));
            }
        }
    }

    // Print generated scenario comparison
    eprintln!("\n=== Generated Scenario Performance Comparison ===\n");
    eprintln!(
        "{:<22} | {:<12} | {:>14} | {:>14} | {:>8}",
        "Scenario", "Resolution", "Baseline (ms)", "Optimized (ms)", "Speedup"
    );
    eprintln!("{}", "-".repeat(80));
    for (name, res, base, opt) in &gen_pairs {
        let speedup = base / opt;
        eprintln!(
            "{:<22} | {:<12} | {:>14.3} | {:>14.3} | {:>7.2}x",
            name, res, base, opt, speedup
        );
    }

    // Print existing SVG summary
    eprintln!(
        "\n=== Existing Test SVGs ({} files, 1x & 3x scale) ===\n",
        exist_1x_count.max(exist_3x_count)
    );
    if exist_1x_count > 0 {
        eprintln!("Total baseline (1x):  {:>10.3} ms", exist_1x_base);
        eprintln!("Total optimized (1x): {:>10.3} ms", exist_1x_opt);
        eprintln!(
            "Overall speedup (1x): {:>10.2}x",
            if exist_1x_opt > 0.0 {
                exist_1x_base / exist_1x_opt
            } else {
                0.0
            }
        );
    }
    if exist_3x_count > 0 {
        eprintln!("Total baseline (3x):  {:>10.3} ms", exist_3x_base);
        eprintln!("Total optimized (3x): {:>10.3} ms", exist_3x_opt);
        eprintln!(
            "Overall speedup (3x): {:>10.2}x",
            if exist_3x_opt > 0.0 {
                exist_3x_base / exist_3x_opt
            } else {
                0.0
            }
        );
    }

    // Print per-filter summary
    eprintln!("\n=== Per-Filter Type Summary ===\n");
    eprintln!(
        "{:<25} | {:>8} | {:>10} | {:>10}",
        "Filter", "Tests", "Avg Speed", "Max Speed"
    );
    eprintln!("{}", "-".repeat(62));
    let mut filter_names: Vec<&String> = filter_stats.keys().collect();
    filter_names.sort();
    for name in filter_names {
        let stats = &filter_stats[name];
        let avg: f64 = stats.speedups.iter().sum::<f64>() / stats.speedups.len() as f64;
        let max: f64 = stats
            .speedups
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "{:<25} | {:>8} | {:>9.2}x | {:>9.2}x",
            name, stats.count, avg, max
        );
    }

    // Print regressions
    eprintln!("\n=== Regressions (optimized >5% slower) ===\n");
    if regressions.is_empty() {
        eprintln!("(none)");
    } else {
        for (name, res, base, opt) in &regressions {
            let speedup = base / opt;
            eprintln!(
                "{} @ {} : {:.3}ms -> {:.3}ms ({:.2}x)",
                name, res, base, opt, speedup
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Load a filter set from a TSV file: each line is "name\tresolution".
fn load_only_filter(path: &Path) -> HashSet<(String, String)> {
    let file = std::fs::File::open(path).expect("Failed to open --only file");
    let reader = std::io::BufReader::new(file);
    let mut set = HashSet::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            set.insert((parts[0].to_string(), parts[1].to_string()));
        }
    }
    set
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse CLI arguments
    let mut compare_path: Option<PathBuf> = None;
    let mut only_path: Option<PathBuf> = None;
    let mut output_tsv_path: Option<PathBuf> = None;
    let mut num_threads: usize = 1;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--compare" => {
                compare_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--only" => {
                only_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--output-tsv" => {
                output_tsv_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--threads" => {
                num_threads = args[i + 1].parse().expect("Invalid --threads value");
                i += 2;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                eprintln!(
                    "Usage: bench_e2e [--compare <baseline.tsv>] [--output-tsv <file>] \
                     [--only <cases.tsv>] [--threads N]"
                );
                std::process::exit(1);
            }
        }
    }

    let only_filter = only_path.as_deref().map(load_only_filter);

    eprintln!("Generating test cases...");
    let generated = generate_all_cases();
    eprintln!("  {} generated scenarios", generated.len());

    eprintln!("Collecting existing filter test SVGs...");
    let existing = collect_existing_filter_svgs();
    eprintln!("  {} existing test cases (1x + 3x)", existing.len());

    let mut all_cases = generated;
    all_cases.extend(existing);

    // Apply --only filter if specified
    if let Some(ref filter_set) = only_filter {
        let before = all_cases.len();
        all_cases.retain(|c| {
            let res = format!("{}x{}", c.width, c.height);
            filter_set.contains(&(c.name.clone(), res))
        });
        eprintln!("  --only filter: {} -> {} cases", before, all_cases.len());
    }

    eprintln!(
        "Total: {} test cases, {} thread(s), resolution-scaled iterations",
        all_cases.len(),
        num_threads,
    );
    eprintln!("Running benchmarks...");

    let results = run_bench(all_cases, num_threads);

    // Output TSV to file or stdout
    if let Some(ref path) = output_tsv_path {
        let mut file = std::fs::File::create(path).expect("Failed to create output TSV file");
        write_tsv(&results, &mut file);
        eprintln!("TSV results written to {}", path.display());
    } else {
        write_tsv(&results, &mut std::io::stdout());
    }

    // If comparing, load baseline and print comparison to stderr
    if let Some(path) = compare_path {
        let baseline = load_tsv(&path);
        compare_results(&baseline, &results);
    }

    eprintln!("Done.");
}
