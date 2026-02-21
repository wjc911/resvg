// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for `feConvolveMatrix` filter primitive.
//!
//! Usage: cargo run --release --example bench_convolve_matrix

use std::time::Instant;

fn main() {
    let resolutions: &[(u32, u32)] = &[(64, 64), (256, 256), (1024, 1024), (4096, 4096)];
    let kernel_sizes: &[u32] = &[3, 5, 7, 9];

    println!(
        "{:<16} {:<12} {:<15} {:<15}",
        "Resolution", "Kernel", "Time (ms)", "Mpix/s"
    );
    println!("{}", "-".repeat(60));

    for &(w, h) in resolutions {
        for &k in kernel_sizes {
            let svg = generate_svg(w, h, k);
            let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
            let mut pixmap = tiny_skia::Pixmap::new(w, h).unwrap();

            // Warmup
            resvg::render(
                &tree,
                tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );

            // Benchmark
            let iterations = pick_iterations(w, h, k);
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
                format!("{}x{}", k, k),
                ms,
                mpix,
            );
        }
    }
}

fn pick_iterations(w: u32, h: u32, k: u32) -> u32 {
    let pixels = w as u64 * h as u64;
    let ops = pixels * k as u64 * k as u64;
    // Target ~100ms total benchmark time
    let iters = (100_000_000u64 / ops.max(1)).max(1).min(1000);
    iters as u32
}

fn generate_svg(width: u32, height: u32, kernel_size: u32) -> String {
    let n = kernel_size * kernel_size;
    // Use a simple averaging kernel
    let kernel_values: Vec<String> = (0..n).map(|_| "1".to_string()).collect();
    let kernel_str = kernel_values.join(" ");
    let divisor = n as f32;

    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <filter id="conv" x="0" y="0" width="100%" height="100%">
      <feConvolveMatrix order="{k}" kernelMatrix="{kernel}" divisor="{divisor}" edgeMode="duplicate"/>
    </filter>
  </defs>
  <rect width="{w}" height="{h}" fill="red" filter="url(#conv)"/>
</svg>"#,
        w = width,
        h = height,
        k = kernel_size,
        kernel = kernel_str,
        divisor = divisor,
    )
}
