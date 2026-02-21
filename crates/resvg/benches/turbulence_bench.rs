// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for feTurbulence filter.
//!
//! Run with: cargo bench --bench turbulence_bench -p resvg

use std::hint::black_box;
use std::time::Instant;

// Re-export the apply function via the public crate interface would require
// exposing internals. Instead, we duplicate the minimal code needed to call
// the turbulence filter directly. This benchmark file measures wall-clock
// time using std::time::Instant (no external benchmark framework dependency).

// We call into resvg's turbulence filter indirectly by constructing the
// necessary types. Since turbulence::apply is pub(crate), we use a small
// SVG document and render it to measure performance.

fn bench_turbulence(
    width: u32,
    height: u32,
    num_octaves: u32,
    stitch_tiles: bool,
    fractal_noise: bool,
    iterations: u32,
) -> std::time::Duration {
    // Create an SVG with feTurbulence
    let stitch_str = if stitch_tiles { "stitch" } else { "noStitch" };
    let type_str = if fractal_noise {
        "fractalNoise"
    } else {
        "turbulence"
    };

    let svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">
  <defs>
    <filter id="turb" x="0" y="0" width="100%" height="100%">
      <feTurbulence baseFrequency="0.05" numOctaves="{}" seed="42"
                    stitchTiles="{}" type="{}"/>
    </filter>
  </defs>
  <rect width="100%" height="100%" filter="url(#turb)"/>
</svg>"#,
        width, height, num_octaves, stitch_str, type_str
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
    println!("feTurbulence Benchmark");
    println!("======================\n");

    let resolutions: &[(u32, u32)] = &[(64, 64), (256, 256), (1024, 1024), (4096, 4096)];
    let octave_counts: &[u32] = &[1, 2, 4, 8];

    // Print header
    println!(
        "{:<12} {:<8} {:<8} {:<10} {:<12} {:<14}",
        "Resolution", "Octaves", "Stitch", "Mode", "Time (ms)", "Mpx/s"
    );
    println!("{}", "-".repeat(72));

    for &(w, h) in resolutions {
        let pixels = w as f64 * h as f64;
        // Adjust iterations to keep total time reasonable
        let base_iters = if pixels > 1_000_000.0 {
            2
        } else if pixels > 100_000.0 {
            10
        } else {
            50
        };

        for &octaves in octave_counts {
            for &stitch in &[false, true] {
                for &fractal in &[false, true] {
                    let iters = base_iters;
                    let elapsed = bench_turbulence(w, h, octaves, stitch, fractal, iters);
                    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iters as f64;
                    let mpx_per_sec = pixels / (ms_per_iter / 1000.0) / 1_000_000.0;

                    let mode = if fractal { "fractal" } else { "turbulence" };
                    let stitch_str = if stitch { "yes" } else { "no" };

                    println!(
                        "{:<12} {:<8} {:<8} {:<10} {:<12.3} {:<14.2}",
                        format!("{}x{}", w, h),
                        octaves,
                        stitch_str,
                        mode,
                        ms_per_iter,
                        mpx_per_sec
                    );
                }
            }
        }
        println!();
    }
}
