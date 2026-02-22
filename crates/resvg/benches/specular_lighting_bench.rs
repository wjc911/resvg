// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for feSpecularLighting filter primitive with real-world usage patterns.
//!
//! Covers the 3D-button size range with MDN canonical parameters.
//! PointLight is the primary (most common) light source.
//!
//! Run with: cargo bench --bench specular_lighting_bench -p resvg

use std::hint::black_box;
use std::time::Instant;

fn make_svg(width: u32, height: u32, exponent: f32, light_type: &str) -> String {
    let light_element = match light_type {
        "distant" => r#"<feDistantLight azimuth="45" elevation="45"/>"#.to_string(),
        "point" => {
            r#"<fePointLight x="50" y="100" z="200"/>"#.to_string()
        }
        "spot" => {
            r#"<feSpotLight x="50" y="100" z="200" pointsAtX="100" pointsAtY="100" pointsAtZ="0" specularExponent="8" limitingConeAngle="30"/>"#.to_string()
        }
        _ => unreachable!(),
    };

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <filter id="f">
      <feSpecularLighting in="SourceGraphic" surfaceScale="5"
        specularConstant="0.75" specularExponent="{exp}"
        lighting-color="white">
        {light}
      </feSpecularLighting>
    </filter>
  </defs>
  <rect width="{w}" height="{h}" fill="gray" filter="url(#f)"/>
</svg>"##,
        w = width,
        h = height,
        exp = exponent,
        light = light_element,
    )
}

fn main() {
    println!("feSpecularLighting Benchmark — Real-World Usage Patterns");
    println!("=========================================================\n");

    let resolutions: &[(u32, u32)] = &[
        (24, 24),
        (48, 48),
        (96, 96),
        (200, 150),
        (400, 300),
    ];
    let exponents: &[f32] = &[1.0, 5.0, 12.0, 20.0, 128.0];
    // PointLight listed first as it is the most common light source
    let light_types: &[&str] = &["point", "distant", "spot"];

    println!(
        "{:<12} {:<10} {:<10} {:<12} {:<14} {:<10}",
        "Resolution", "Light", "Exponent", "Time (ms)", "Mpix/s", "Iters"
    );
    println!("{}", "-".repeat(70));

    for &(width, height) in resolutions {
        for &light_type in light_types {
            for &exponent in exponents {
                let svg = make_svg(width, height, exponent, light_type);
                let tree = resvg::usvg::Tree::from_str(
                    &svg,
                    &resvg::usvg::Options::default(),
                )
                .unwrap();

                // Warmup
                for _ in 0..2 {
                    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height).unwrap();
                    resvg::render(
                        &tree,
                        resvg::tiny_skia::Transform::identity(),
                        &mut pixmap.as_mut(),
                    );
                }

                // Dynamic iteration count: target ~2 seconds per measurement
                let probe_start = Instant::now();
                {
                    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height).unwrap();
                    resvg::render(
                        &tree,
                        resvg::tiny_skia::Transform::identity(),
                        &mut pixmap.as_mut(),
                    );
                }
                let probe_ms = probe_start.elapsed().as_secs_f64() * 1000.0;
                let iterations = ((2000.0 / probe_ms).ceil() as u32).max(2).min(2000);

                let start = Instant::now();
                for _ in 0..iterations {
                    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height).unwrap();
                    resvg::render(
                        &tree,
                        resvg::tiny_skia::Transform::identity(),
                        &mut pixmap.as_mut(),
                    );
                    black_box(&pixmap);
                }
                let elapsed = start.elapsed();

                let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
                let pixels = width as f64 * height as f64;
                let mpix_per_sec = pixels / (ms_per_iter / 1000.0) / 1_000_000.0;

                println!(
                    "{:<12} {:<10} {:<10} {:<12.3} {:<14.2} {:<10}",
                    format!("{}x{}", width, height),
                    light_type,
                    exponent,
                    ms_per_iter,
                    mpix_per_sec,
                    iterations
                );
            }
        }
        println!();
    }
}
