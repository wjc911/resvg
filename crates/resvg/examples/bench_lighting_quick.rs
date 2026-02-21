// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Quick comparison benchmark for feDiffuseLighting and feSpecularLighting.
//! Runs a representative subset of configurations for fast regression detection.
//!
//! Usage: cargo run --release --example bench_lighting_quick

use std::time::Instant;

struct BenchCase {
    name: &'static str,
    svg: String,
    width: u32,
    height: u32,
}

fn make_diffuse_svg(w: u32, h: u32, light: &str, dc: f32, ss: f32) -> String {
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="white" stop-opacity="0"/>
      <stop offset="1" stop-color="white" stop-opacity="1"/>
    </linearGradient>
    <filter id="f" x="0" y="0" width="100%" height="100%">
      <feDiffuseLighting surfaceScale="{ss}" diffuseConstant="{dc}" lighting-color="white">
        {light}
      </feDiffuseLighting>
    </filter>
  </defs>
  <g filter="url(#f)">
    <rect width="{w}" height="{h}" fill="url(#g)"/>
  </g>
</svg>"##,
    )
}

fn make_specular_svg(w: u32, h: u32, light: &str, se: f32, sc: f32, ss: f32) -> String {
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="white" stop-opacity="0"/>
      <stop offset="1" stop-color="white" stop-opacity="1"/>
    </linearGradient>
    <filter id="f" x="0" y="0" width="100%" height="100%">
      <feSpecularLighting surfaceScale="{ss}" specularConstant="{sc}" specularExponent="{se}" lighting-color="white">
        {light}
      </feSpecularLighting>
    </filter>
  </defs>
  <g filter="url(#f)">
    <rect width="{w}" height="{h}" fill="url(#g)"/>
  </g>
</svg>"##,
    )
}

fn pick_iters(w: u32, h: u32) -> u32 {
    let p = w as u64 * h as u64;
    if p <= 256 { 5000 }
    else if p <= 4096 { 2000 }
    else if p <= 16384 { 500 }
    else if p <= 65536 { 200 }
    else if p <= 262144 { 50 }
    else { 10 }
}

fn bench(case: &BenchCase) -> f64 {
    let tree = usvg::Tree::from_str(&case.svg, &usvg::Options::default()).unwrap();
    let mut pixmap = tiny_skia::Pixmap::new(case.width, case.height).unwrap();

    // warmup
    for _ in 0..3 {
        resvg::render(&tree, tiny_skia::Transform::identity(), &mut pixmap.as_mut());
    }

    let iters = pick_iters(case.width, case.height);
    let mut best = f64::MAX;

    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..iters {
            resvg::render(&tree, tiny_skia::Transform::identity(), &mut pixmap.as_mut());
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
        if us < best { best = us; }
    }
    best
}

fn main() {
    let distant = r#"<feDistantLight azimuth="45" elevation="55"/>"#;
    let point = r#"<fePointLight x="150" y="60" z="200"/>"#;
    let spot = r#"<feSpotLight x="150" y="60" z="200" pointsAtX="100" pointsAtY="100" pointsAtZ="0" specularExponent="8" limitingConeAngle="30"/>"#;

    let sizes: &[(u32, u32)] = &[
        (4, 4), (16, 16), (32, 32), (64, 64),
        (127, 127), (128, 128), (129, 129),
        (256, 256), (512, 512), (1024, 1024),
    ];

    // ==================== feDiffuseLighting ====================
    println!("feDiffuseLighting Quick Benchmark");
    println!("{:<14} {:<10} {:<16} {:>12} {:>12}", "Size", "Light", "Params", "Time(us)", "Mpix/s");
    println!("{}", "-".repeat(70));

    let lights: &[(&str, &str)] = &[("distant", distant), ("point", point), ("spot", spot)];
    let diff_params: &[(f32, f32, &str)] = &[
        (1.0, 1.0, "dc=1,ss=1"),
        (1.0, 5.0, "dc=1,ss=5"),
        (2.0, 10.0, "dc=2,ss=10"),
    ];

    for &(w, h) in sizes {
        for &(lname, lxml) in lights {
            for &(dc, ss, pdesc) in diff_params {
                let svg = make_diffuse_svg(w, h, lxml, dc, ss);
                let c = BenchCase { name: "diffuse", svg, width: w, height: h };
                let us = bench(&c);
                let mpix = (w as f64 * h as f64) / us;
                println!("{:<14} {:<10} {:<16} {:>12.1} {:>12.2}", format!("{}x{}", w, h), lname, pdesc, us, mpix);
            }
        }
        println!("{}", "-".repeat(70));
    }

    // ==================== feSpecularLighting ====================
    println!("\nfeSpecularLighting Quick Benchmark");
    println!("{:<14} {:<10} {:<16} {:>12} {:>12}", "Size", "Light", "Params", "Time(us)", "Mpix/s");
    println!("{}", "-".repeat(70));

    let spec_params: &[(f32, f32, f32, &str)] = &[
        (1.0, 1.0, 1.0, "se=1,sc=1,ss=1"),
        (20.0, 1.0, 5.0, "se=20,sc=1,ss=5"),
        (128.0, 0.5, 1.0, "se=128,sc=.5,ss=1"),
    ];

    for &(w, h) in sizes {
        for &(lname, lxml) in lights {
            for &(se, sc, ss, pdesc) in spec_params {
                let svg = make_specular_svg(w, h, lxml, se, sc, ss);
                let c = BenchCase { name: "specular", svg, width: w, height: h };
                let us = bench(&c);
                let mpix = (w as f64 * h as f64) / us;
                println!("{:<14} {:<10} {:<16} {:>12.1} {:>12.2}", format!("{}x{}", w, h), lname, pdesc, us, mpix);
            }
        }
        println!("{}", "-".repeat(70));
    }
}
