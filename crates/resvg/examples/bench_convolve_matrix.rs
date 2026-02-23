// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for `feConvolveMatrix` filter primitive with real-world usage patterns.
//!
//! Scenarios are based on common SVG convolution use cases:
//! - 3×3 photo sharpen (most common by far)
//! - 3×3 emboss effect (UI/text styling)
//! - 3×3 edge detection / Laplacian
//! - 3×3 Gaussian approximation (separable kernel)
//! - 5×5 unsharp mask (rare but realistic stress test)
//! - 1×1 identity (trivial / pass-through)
//!
//! Usage: cargo run --release --example bench_convolve_matrix

use std::time::Instant;

struct Scenario {
    name: &'static str,
    kernel: &'static [f32],
    order: u32,
    divisor: f32,
    sizes: &'static [(u32, u32)],
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "Photo Sharpen 3x3",
        kernel: &[0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
        order: 3,
        divisor: 1.0,
        sizes: &[
            (200, 150),
            (400, 300),
            (600, 400),
            (800, 600),
            (1024, 768),
            (1500, 1000),
        ],
    },
    Scenario {
        name: "Emboss 3x3",
        kernel: &[-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0],
        order: 3,
        divisor: 1.0,
        sizes: &[(200, 150), (400, 300), (600, 400)],
    },
    Scenario {
        name: "Edge Detect 3x3",
        kernel: &[0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
        order: 3,
        divisor: 1.0,
        sizes: &[(400, 300), (600, 400), (800, 600), (1024, 768)],
    },
    Scenario {
        name: "Gaussian 3x3",
        kernel: &[1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
        order: 3,
        divisor: 16.0,
        sizes: &[
            (400, 300),
            (600, 400),
            (800, 600),
            (1024, 768),
            (1500, 1000),
        ],
    },
    Scenario {
        name: "Unsharp 5x5",
        kernel: &[
            1.0, 4.0, 6.0, 4.0, 1.0, 4.0, 16.0, 24.0, 16.0, 4.0, 6.0, 24.0, -476.0, 24.0, 6.0, 4.0,
            16.0, 24.0, 16.0, 4.0, 1.0, 4.0, 6.0, 4.0, 1.0,
        ],
        order: 5,
        divisor: -256.0,
        sizes: &[(400, 300), (600, 400), (800, 600), (1024, 768)],
    },
    Scenario {
        name: "Trivial 1x1",
        kernel: &[1.0],
        order: 1,
        divisor: 1.0,
        sizes: &[(200, 150), (400, 300), (800, 600), (1500, 1000)],
    },
];

fn main() {
    println!(
        "{:<22} {:<14} {:>10} {:>10}",
        "Scenario", "Size", "ms/iter", "Mpix/s"
    );
    println!("{}", "-".repeat(58));

    for scenario in SCENARIOS {
        for &(w, h) in scenario.sizes {
            let svg = generate_svg(w, h, scenario);
            let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
            let mut pixmap = tiny_skia::Pixmap::new(w, h).unwrap();

            // Warmup
            resvg::render(
                &tree,
                tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );

            // Calibrate: run a few iterations to estimate cost, then pick count
            let iterations = calibrate(|| {
                resvg::render(
                    &tree,
                    tiny_skia::Transform::identity(),
                    &mut pixmap.as_mut(),
                );
            });

            // Timed run
            let start = Instant::now();
            for _ in 0..iterations {
                resvg::render(
                    &tree,
                    tiny_skia::Transform::identity(),
                    &mut pixmap.as_mut(),
                );
            }
            let elapsed = start.elapsed();

            let ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
            let mpix = (w as f64 * h as f64) / (ms * 1000.0);

            println!(
                "{:<22} {:>5}x{:<5} {:>10.3} {:>10.2}",
                scenario.name, w, h, ms, mpix,
            );
        }
        println!();
    }
}

/// Run `f` a few times to estimate cost, then return an iteration count
/// that targets ~200ms total benchmark time.
fn calibrate<F: FnMut()>(mut f: F) -> u32 {
    // Run 2 iterations to estimate per-call cost
    let probe = Instant::now();
    f();
    f();
    let probe_ns = probe.elapsed().as_nanos() as f64 / 2.0;

    // Target 200ms total
    let target_ns = 200_000_000.0;
    let n = (target_ns / probe_ns).round() as u32;
    n.max(2).min(5000)
}

fn generate_svg(width: u32, height: u32, scenario: &Scenario) -> String {
    let kernel_str: String = scenario
        .kernel
        .iter()
        .map(|v| {
            if *v == v.trunc() {
                format!("{}", *v as i32)
            } else {
                format!("{}", v)
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <filter id="conv" x="0" y="0" width="100%" height="100%">
      <feConvolveMatrix order="{order}" kernelMatrix="{kernel}" divisor="{divisor}" edgeMode="duplicate"/>
    </filter>
  </defs>
  <rect width="{w}" height="{h}" fill="red" filter="url(#conv)"/>
</svg>"#,
        w = width,
        h = height,
        order = scenario.order,
        kernel = kernel_str,
        divisor = scenario.divisor,
    )
}
