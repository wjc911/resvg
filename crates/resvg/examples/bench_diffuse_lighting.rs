// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for `feDiffuseLighting` filter primitive.
//!
//! Usage: cargo run --release --example bench_diffuse_lighting

use std::time::Instant;

fn main() {
    let resolutions: &[(u32, u32)] = &[(64, 64), (256, 256), (1024, 1024), (4096, 4096)];
    let light_types: &[(&str, &str)] = &[
        (
            "distant",
            r#"<feDistantLight azimuth="45" elevation="55"/>"#,
        ),
        ("point", r#"<fePointLight x="150" y="60" z="200"/>"#),
        (
            "spot",
            r#"<feSpotLight x="150" y="60" z="200" pointsAtX="100" pointsAtY="100" pointsAtZ="0" specularExponent="8" limitingConeAngle="30"/>"#,
        ),
    ];

    println!(
        "{:<16} {:<12} {:<15} {:<15}",
        "Resolution", "Light", "Time (ms)", "Mpix/s"
    );
    println!("{}", "-".repeat(60));

    for &(w, h) in resolutions {
        for &(name, light_elem) in light_types {
            let svg = generate_svg(w, h, light_elem);
            let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
            let mut pixmap = tiny_skia::Pixmap::new(w, h).unwrap();

            // Warmup
            resvg::render(
                &tree,
                tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );

            // Benchmark
            let iterations = pick_iterations(w, h);
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
                "{:<16} {:<12} {:<15.3} {:<15.2}",
                format!("{}x{}", w, h),
                name,
                ms,
                mpix,
            );
        }
    }
}

fn pick_iterations(w: u32, h: u32) -> u32 {
    let pixels = w as u64 * h as u64;
    // Target ~200ms total benchmark time
    let iters = (200_000_000u64 / pixels.max(1)).max(1).min(1000);
    iters as u32
}

fn generate_svg(width: u32, height: u32, light_element: &str) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <filter id="dl" x="0" y="0" width="100%" height="100%">
      <feDiffuseLighting surfaceScale="5" diffuseConstant="1" lighting-color="white">
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
