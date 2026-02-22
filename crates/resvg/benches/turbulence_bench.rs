// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for feTurbulence filter with real-world usage patterns.
//!
//! Run with: cargo bench --bench turbulence_bench -p resvg

use std::hint::black_box;
use std::time::Instant;

struct Scenario {
    name: &'static str,
    base_frequency: &'static str,
    num_octaves: u32,
    turbulence_type: &'static str,
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "Fine Noise",
        base_frequency: "0.05",
        num_octaves: 2,
        turbulence_type: "turbulence",
    },
    Scenario {
        name: "Cloud Texture",
        base_frequency: "0.01",
        num_octaves: 3,
        turbulence_type: "fractalNoise",
    },
    Scenario {
        name: "Paper Texture",
        base_frequency: "0.04",
        num_octaves: 5,
        turbulence_type: "fractalNoise",
    },
    Scenario {
        name: "Directional Grain",
        base_frequency: "0.1 0.01",
        num_octaves: 2,
        turbulence_type: "fractalNoise",
    },
    Scenario {
        name: "High Frequency",
        base_frequency: "0.2",
        num_octaves: 1,
        turbulence_type: "turbulence",
    },
];

const RESOLUTIONS: &[(u32, u32)] = &[
    (200, 150),
    (400, 300),
    (600, 400),
    (800, 600),
    (1024, 768),
];

fn bench_turbulence(
    width: u32,
    height: u32,
    scenario: &Scenario,
    iterations: u32,
) -> std::time::Duration {
    let svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">
  <defs>
    <filter id="turb" x="0" y="0" width="100%" height="100%">
      <feTurbulence baseFrequency="{}" numOctaves="{}" seed="42"
                    stitchTiles="noStitch" type="{}"/>
    </filter>
  </defs>
  <rect width="100%" height="100%" filter="url(#turb)"/>
</svg>"#,
        width, height, scenario.base_frequency, scenario.num_octaves, scenario.turbulence_type
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
    println!("feTurbulence Benchmark — Real-World Usage Patterns");
    println!("===================================================\n");

    // Print header
    println!(
        "{:<20} {:<12} {:<12} {:<14} {:<10}",
        "Scenario", "Resolution", "Time (ms)", "Mpix/s", "Iters"
    );
    println!("{}", "-".repeat(70));

    for scenario in SCENARIOS {
        for &(w, h) in RESOLUTIONS {
            let pixels = w as f64 * h as f64;

            // Dynamic iteration count: target ~2 seconds per measurement
            let probe_dur = bench_turbulence(w, h, scenario, 1);
            let probe_ms = probe_dur.as_secs_f64() * 1000.0;
            let iterations = ((2000.0 / probe_ms).ceil() as u32).max(2).min(500);

            let elapsed = bench_turbulence(w, h, scenario, iterations);
            let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
            let mpx_per_sec = pixels / (ms_per_iter / 1000.0) / 1_000_000.0;

            println!(
                "{:<20} {:<12} {:<12.3} {:<14.2} {:<10}",
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
