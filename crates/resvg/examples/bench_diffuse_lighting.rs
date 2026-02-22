// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for `feDiffuseLighting` filter primitive with real-world usage patterns.
//!
//! Covers the 3D-button size range with realistic parameters (MDN canonical values)
//! and prioritizes PointLight as the most common light source.
//!
//! Usage: cargo run --release --example bench_diffuse_lighting -p resvg

use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("feDiffuseLighting Benchmark — Real-World Usage Patterns");
    println!("========================================================\n");

    let resolutions: &[(u32, u32)] = &[
        (24, 24),
        (48, 48),
        (96, 96),
        (128, 128),
        (200, 150),
        (400, 300),
    ];

    // PointLight listed first as it is the most common light source
    let light_types: &[(&str, &str)] = &[
        (
            "point",
            r#"<fePointLight x="50" y="100" z="200"/>"#,
        ),
        (
            "distant",
            r#"<feDistantLight azimuth="45" elevation="55"/>"#,
        ),
        (
            "spot",
            r#"<feSpotLight x="50" y="100" z="200" pointsAtX="100" pointsAtY="100" pointsAtZ="0" specularExponent="8" limitingConeAngle="30"/>"#,
        ),
    ];

    println!(
        "{:<12} {:<12} {:<12} {:<14} {:<10}",
        "Resolution", "Light", "Time (ms)", "Mpix/s", "Iters"
    );
    println!("{}", "-".repeat(62));

    for &(w, h) in resolutions {
        for &(name, light_elem) in light_types {
            let svg = generate_svg(w, h, light_elem);
            let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();

            // Warmup
            for _ in 0..2 {
                let mut pixmap = tiny_skia::Pixmap::new(w, h).unwrap();
                resvg::render(
                    &tree,
                    tiny_skia::Transform::identity(),
                    &mut pixmap.as_mut(),
                );
            }

            // Dynamic iteration count: target ~2 seconds per measurement
            let probe_start = Instant::now();
            {
                let mut pixmap = tiny_skia::Pixmap::new(w, h).unwrap();
                resvg::render(
                    &tree,
                    tiny_skia::Transform::identity(),
                    &mut pixmap.as_mut(),
                );
            }
            let probe_ms = probe_start.elapsed().as_secs_f64() * 1000.0;
            let iterations = ((2000.0 / probe_ms).ceil() as u32).max(2).min(2000);

            let start = Instant::now();
            for _ in 0..iterations {
                let mut pixmap = tiny_skia::Pixmap::new(w, h).unwrap();
                resvg::render(
                    &tree,
                    tiny_skia::Transform::identity(),
                    &mut pixmap.as_mut(),
                );
                black_box(&pixmap);
            }
            let elapsed = start.elapsed();

            let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
            let pixels = w as f64 * h as f64;
            let mpix_per_sec = pixels / (ms_per_iter / 1000.0) / 1_000_000.0;

            println!(
                "{:<12} {:<12} {:<12.3} {:<14.2} {:<10}",
                format!("{}x{}", w, h),
                name,
                ms_per_iter,
                mpix_per_sec,
                iterations
            );
        }
        println!();
    }
}

fn generate_svg(width: u32, height: u32, light_element: &str) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <filter id="dl" x="0" y="0" width="100%" height="100%">
      <feDiffuseLighting surfaceScale="5" diffuseConstant="0.75" lighting-color="white">
        {light}
      </feDiffuseLighting>
    </filter>
  </defs>
  <rect width="{w}" height="{h}" fill="red" filter="url(#dl)"/>
</svg>"#,
        w = width,
        h = height,
        light = light_element,
    )
}
