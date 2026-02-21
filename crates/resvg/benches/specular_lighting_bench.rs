// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for feSpecularLighting filter primitive.
//!
//! Run with: cargo bench --bench specular_lighting_bench -p resvg

use std::time::Instant;

fn make_svg(width: u32, height: u32, exponent: f32, light_type: &str) -> String {
    let light_element = match light_type {
        "distant" => r#"<feDistantLight azimuth="45" elevation="45"/>"#.to_string(),
        "point" => {
            format!(
                r#"<fePointLight x="{}" y="{}" z="100"/>"#,
                width / 2,
                height / 2
            )
        }
        "spot" => {
            format!(
                r#"<feSpotLight x="{}" y="{}" z="100" pointsAtX="{}" pointsAtY="{}" pointsAtZ="0"/>"#,
                width / 2,
                height / 2,
                width,
                height
            )
        }
        _ => unreachable!(),
    };

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <filter id="f">
      <feSpecularLighting in="SourceGraphic" surfaceScale="1"
        specularConstant="1" specularExponent="{exp}"
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

fn bench_one(
    label: &str,
    svg_str: &str,
    warmup_iters: u32,
    bench_iters: u32,
    width: u32,
    height: u32,
) {
    let tree = resvg::usvg::Tree::from_str(svg_str, &resvg::usvg::Options::default()).unwrap();

    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height).unwrap();

    // warmup
    for _ in 0..warmup_iters {
        pixmap.fill(resvg::tiny_skia::Color::TRANSPARENT);
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
    }

    // benchmark
    let start = Instant::now();
    for _ in 0..bench_iters {
        pixmap.fill(resvg::tiny_skia::Color::TRANSPARENT);
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
    }
    let elapsed = start.elapsed();

    let per_iter = elapsed / bench_iters;
    let pixels = (width as u64) * (height as u64);
    let mpixels_per_sec =
        (pixels as f64 * bench_iters as f64) / elapsed.as_secs_f64() / 1_000_000.0;

    println!(
        "{:<60} {:>10.3} ms/iter  ({:.1} Mpx/s)",
        label,
        per_iter.as_secs_f64() * 1000.0,
        mpixels_per_sec,
    );
}

fn main() {
    let resolutions: &[(u32, u32)] = &[(64, 64), (256, 256), (1024, 1024), (4096, 4096)];
    let exponents: &[f32] = &[1.0, 5.0, 20.0, 128.0];
    let light_types: &[&str] = &["distant", "point", "spot"];

    println!("feSpecularLighting Benchmark");
    println!("============================");
    println!();

    for &(width, height) in resolutions {
        println!("--- Resolution: {}x{} ---", width, height);

        let (warmup, iters) = if width <= 256 {
            (10, 100)
        } else if width <= 1024 {
            (3, 20)
        } else {
            (1, 5)
        };

        for &exponent in exponents {
            for &light_type in light_types {
                let svg = make_svg(width, height, exponent, light_type);
                let label = format!(
                    "specular exp={:<5} light={:<8} {}x{}",
                    exponent, light_type, width, height
                );
                bench_one(&label, &svg, warmup, iters, width, height);
            }
        }
        println!();
    }
}
