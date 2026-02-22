// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for feComponentTransfer filter with real-world usage patterns.
//!
//! Scenarios are based on common SVG component transfer use cases:
//! - Gamma correction (sRGB to linear) — the most common use case
//! - Inverse gamma (linear to sRGB)
//! - Duotone table (2-value color mapping)
//! - Posterization (discrete 4-level quantization)
//! - Contrast boost (linear slope/intercept)
//! - Heavy posterization (discrete 16-level quantization)
//!
//! Usage: cargo run --release --example bench_component_transfer -p resvg

use std::hint::black_box;
use std::time::Instant;

struct Scenario {
    name: &'static str,
    filter_body: &'static str,
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "Gamma Correction (sRGB->linear)",
        filter_body: r#"<feComponentTransfer>
        <feFuncR type="gamma" amplitude="1" exponent="2.2" offset="0"/>
        <feFuncG type="gamma" amplitude="1" exponent="2.2" offset="0"/>
        <feFuncB type="gamma" amplitude="1" exponent="2.2" offset="0"/>
      </feComponentTransfer>"#,
    },
    Scenario {
        name: "Inverse Gamma (linear->sRGB)",
        filter_body: r#"<feComponentTransfer>
        <feFuncR type="gamma" amplitude="1" exponent="0.4545" offset="0"/>
        <feFuncG type="gamma" amplitude="1" exponent="0.4545" offset="0"/>
        <feFuncB type="gamma" amplitude="1" exponent="0.4545" offset="0"/>
      </feComponentTransfer>"#,
    },
    Scenario {
        name: "Duotone Table",
        filter_body: r#"<feComponentTransfer>
        <feFuncR type="table" tableValues="0.2 0.8"/>
        <feFuncG type="table" tableValues="0.1 0.6"/>
        <feFuncB type="table" tableValues="0.3 0.9"/>
      </feComponentTransfer>"#,
    },
    Scenario {
        name: "Posterization (4-level)",
        filter_body: r#"<feComponentTransfer>
        <feFuncR type="discrete" tableValues="0 0.33 0.67 1.0"/>
        <feFuncG type="discrete" tableValues="0 0.33 0.67 1.0"/>
        <feFuncB type="discrete" tableValues="0 0.33 0.67 1.0"/>
      </feComponentTransfer>"#,
    },
    Scenario {
        name: "Contrast Boost",
        filter_body: r#"<feComponentTransfer>
        <feFuncR type="linear" slope="1.5" intercept="-0.1"/>
        <feFuncG type="linear" slope="1.5" intercept="-0.1"/>
        <feFuncB type="linear" slope="1.5" intercept="-0.1"/>
      </feComponentTransfer>"#,
    },
    Scenario {
        name: "Heavy Posterization (16-level)",
        filter_body: r#"<feComponentTransfer>
        <feFuncR type="discrete" tableValues="0.0 0.067 0.133 0.2 0.267 0.333 0.4 0.467 0.533 0.6 0.667 0.733 0.8 0.867 0.933 1.0"/>
        <feFuncG type="discrete" tableValues="0.0 0.067 0.133 0.2 0.267 0.333 0.4 0.467 0.533 0.6 0.667 0.733 0.8 0.867 0.933 1.0"/>
        <feFuncB type="discrete" tableValues="0.0 0.067 0.133 0.2 0.267 0.333 0.4 0.467 0.533 0.6 0.667 0.733 0.8 0.867 0.933 1.0"/>
      </feComponentTransfer>"#,
    },
];

const RESOLUTIONS: &[(u32, u32)] = &[
    (200, 150),
    (400, 300),
    (600, 400),
    (800, 600),
    (1024, 768),
];

fn bench_component_transfer(
    width: u32,
    height: u32,
    scenario: &Scenario,
    iterations: u32,
) -> std::time::Duration {
    let svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">
  <defs>
    <filter id="ct" x="0" y="0" width="100%" height="100%">
      {}
    </filter>
  </defs>
  <rect width="100%" height="100%" fill="red" filter="url(#ct)"/>
</svg>"#,
        width, height, scenario.filter_body
    );

    let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();

    // Warm up
    for _ in 0..2 {
        let mut pixmap = tiny_skia::Pixmap::new(width, height).unwrap();
        resvg::render(&tree, tiny_skia::Transform::default(), &mut pixmap.as_mut());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let mut pixmap = tiny_skia::Pixmap::new(width, height).unwrap();
        resvg::render(&tree, tiny_skia::Transform::default(), &mut pixmap.as_mut());
        black_box(&pixmap);
    }
    start.elapsed()
}

fn main() {
    println!("feComponentTransfer Benchmark — Real-World Usage Patterns");
    println!("==========================================================\n");

    // Print header
    println!(
        "{:<35} {:<12} {:<12} {:<14} {:<10}",
        "Scenario", "Resolution", "Time (ms)", "Mpix/s", "Iters"
    );
    println!("{}", "-".repeat(85));

    for scenario in SCENARIOS {
        for &(w, h) in RESOLUTIONS {
            let pixels = w as f64 * h as f64;

            // Dynamic iteration count: target ~2 seconds per measurement
            let probe_dur = bench_component_transfer(w, h, scenario, 1);
            let probe_ms = probe_dur.as_secs_f64() * 1000.0;
            let iterations = ((2000.0 / probe_ms).ceil() as u32).max(2).min(500);

            let elapsed = bench_component_transfer(w, h, scenario, iterations);
            let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
            let mpx_per_sec = pixels / (ms_per_iter / 1000.0) / 1_000_000.0;

            println!(
                "{:<35} {:<12} {:<12.3} {:<14.2} {:<10}",
                scenario.name,
                format!("{}x{}", w, h),
                ms_per_iter,
                mpx_per_sec,
                iterations
            );
        }
        println!();
    }
}
