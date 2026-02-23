// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for feGaussianBlur with real-world usage patterns.
//!
//! Run with: cargo bench -p resvg --bench blur_bench
//!
//! Code paths covered:
//!   - IIR blur: σ < 2
//!   - Box blur small: σ >= 2, pixel_count <= 250k
//!   - Box blur tiled: σ >= 2, pixel_count > 250k AND radius >= 8
//!   - IIR interleaved: σ < 2, pixel_count > 500k
//!
//! Sigma values mapped to real use cases:
//!   σ=1.5  IIR path (subtle anti-alias, glow)
//!   σ=2    Icon shadow (Material Z1-Z2)
//!   σ=4    Standard UI shadow (MDN canonical, Tailwind backdrop-blur-sm)
//!   σ=8    Tailwind backdrop-blur default, Apple frosted glass low
//!   σ=16   Tailwind backdrop-blur-lg, Apple frosted glass standard
//!   σ=40   Tailwind backdrop-blur-2xl, heavy backdrop blur

#![allow(clippy::needless_range_loop)]

use rgb::RGBA8;
use std::time::Instant;

// ---- Inline the necessary types so the bench can call blur directly ----

struct ImageRefMut<'a> {
    data: &'a mut [RGBA8],
    width: u32,
    height: u32,
}

impl<'a> ImageRefMut<'a> {
    fn new(width: u32, height: u32, data: &'a mut [RGBA8]) -> Self {
        ImageRefMut {
            data,
            width,
            height,
        }
    }
}

// ===========================================================================
// Box blur NAIVE (verbatim copy from original box_blur.rs)
// ===========================================================================
mod box_blur_naive {
    use super::ImageRefMut;
    use rgb::RGBA8;
    use std::cmp;

    const STEPS: usize = 5;

    #[inline(always)]
    pub fn apply(sigma_x: f64, sigma_y: f64, mut src: ImageRefMut) {
        let boxes_horz = create_box_gauss(sigma_x as f32);
        let boxes_vert = create_box_gauss(sigma_y as f32);
        let mut backbuf = src.data.to_vec();
        let mut backbuf = ImageRefMut::new(src.width, src.height, &mut backbuf);
        for (box_size_horz, box_size_vert) in boxes_horz.iter().zip(boxes_vert.iter()) {
            let radius_horz = ((box_size_horz - 1) / 2) as usize;
            let radius_vert = ((box_size_vert - 1) / 2) as usize;
            box_blur_inner(radius_horz, radius_vert, &mut backbuf, &mut src);
        }
    }

    fn create_box_gauss(sigma: f32) -> [i32; STEPS] {
        if sigma > 0.0 {
            let n_float = STEPS as f32;
            let w_ideal = (12.0 * sigma * sigma / n_float).sqrt() + 1.0;
            let mut wl = w_ideal.floor() as i32;
            if wl % 2 == 0 {
                wl -= 1;
            }
            let wu = wl + 2;
            let wl_float = wl as f32;
            let m_ideal = (12.0 * sigma * sigma
                - n_float * wl_float * wl_float
                - 4.0 * n_float * wl_float
                - 3.0 * n_float)
                / (-4.0 * wl_float - 4.0);
            let m = m_ideal.round() as usize;
            let mut sizes = [0; STEPS];
            for i in 0..STEPS {
                if i < m {
                    sizes[i] = wl;
                } else {
                    sizes[i] = wu;
                }
            }
            sizes
        } else {
            [1; STEPS]
        }
    }

    #[inline(always)]
    pub(super) fn box_blur_inner(
        blur_radius_horz: usize,
        blur_radius_vert: usize,
        backbuf: &mut ImageRefMut,
        frontbuf: &mut ImageRefMut,
    ) {
        box_blur_vert(blur_radius_vert, frontbuf, backbuf);
        box_blur_horz(blur_radius_horz, backbuf, frontbuf);
    }

    fn box_blur_vert(blur_radius: usize, backbuf: &ImageRefMut, frontbuf: &mut ImageRefMut) {
        if blur_radius == 0 {
            frontbuf.data.copy_from_slice(backbuf.data);
            return;
        }
        let width = backbuf.width as usize;
        let height = backbuf.height as usize;
        let iarr = 1.0 / (blur_radius + blur_radius + 1) as f32;
        let blur_radius_prev = blur_radius as isize - height as isize;
        let blur_radius_next = blur_radius as isize + 1;
        for i in 0..width {
            let col_start = i;
            let col_end = i + width * (height - 1);
            let mut ti = i;
            let mut li = ti;
            let mut ri = ti + blur_radius * width;
            let fv = RGBA8::default();
            let lv = RGBA8::default();
            let mut val_r = blur_radius_next * (fv.r as isize);
            let mut val_g = blur_radius_next * (fv.g as isize);
            let mut val_b = blur_radius_next * (fv.b as isize);
            let mut val_a = blur_radius_next * (fv.a as isize);
            let get_top = |i: usize| {
                if i < col_start { fv } else { backbuf.data[i] }
            };
            let get_bottom = |i: usize| {
                if i > col_end { lv } else { backbuf.data[i] }
            };
            for j in 0..cmp::min(blur_radius, height) {
                let bb = backbuf.data[ti + j * width];
                val_r += bb.r as isize;
                val_g += bb.g as isize;
                val_b += bb.b as isize;
                val_a += bb.a as isize;
            }
            if blur_radius > height {
                val_r += blur_radius_prev * (lv.r as isize);
                val_g += blur_radius_prev * (lv.g as isize);
                val_b += blur_radius_prev * (lv.b as isize);
                val_a += blur_radius_prev * (lv.a as isize);
            }
            for _ in 0..cmp::min(height, blur_radius + 1) {
                let bb = get_bottom(ri);
                ri += width;
                val_r += sub(bb.r, fv.r);
                val_g += sub(bb.g, fv.g);
                val_b += sub(bb.b, fv.b);
                val_a += sub(bb.a, fv.a);
                frontbuf.data[ti] = RGBA8 {
                    r: round(val_r as f32 * iarr) as u8,
                    g: round(val_g as f32 * iarr) as u8,
                    b: round(val_b as f32 * iarr) as u8,
                    a: round(val_a as f32 * iarr) as u8,
                };
                ti += width;
            }
            if height <= blur_radius {
                continue;
            }
            for _ in (blur_radius + 1)..(height - blur_radius) {
                let bb1 = backbuf.data[ri];
                ri += width;
                let bb2 = backbuf.data[li];
                li += width;
                val_r += sub(bb1.r, bb2.r);
                val_g += sub(bb1.g, bb2.g);
                val_b += sub(bb1.b, bb2.b);
                val_a += sub(bb1.a, bb2.a);
                frontbuf.data[ti] = RGBA8 {
                    r: round(val_r as f32 * iarr) as u8,
                    g: round(val_g as f32 * iarr) as u8,
                    b: round(val_b as f32 * iarr) as u8,
                    a: round(val_a as f32 * iarr) as u8,
                };
                ti += width;
            }
            for _ in 0..cmp::min(height - blur_radius - 1, blur_radius) {
                let bb = get_top(li);
                li += width;
                val_r += sub(lv.r, bb.r);
                val_g += sub(lv.g, bb.g);
                val_b += sub(lv.b, bb.b);
                val_a += sub(lv.a, bb.a);
                frontbuf.data[ti] = RGBA8 {
                    r: round(val_r as f32 * iarr) as u8,
                    g: round(val_g as f32 * iarr) as u8,
                    b: round(val_b as f32 * iarr) as u8,
                    a: round(val_a as f32 * iarr) as u8,
                };
                ti += width;
            }
        }
    }

    fn box_blur_horz(blur_radius: usize, backbuf: &ImageRefMut, frontbuf: &mut ImageRefMut) {
        if blur_radius == 0 {
            frontbuf.data.copy_from_slice(backbuf.data);
            return;
        }
        let width = backbuf.width as usize;
        let height = backbuf.height as usize;
        let iarr = 1.0 / (blur_radius + blur_radius + 1) as f32;
        let blur_radius_prev = blur_radius as isize - width as isize;
        let blur_radius_next = blur_radius as isize + 1;
        for i in 0..height {
            let row_start = i * width;
            let row_end = (i + 1) * width - 1;
            let mut ti = i * width;
            let mut li = ti;
            let mut ri = ti + blur_radius;
            let fv = RGBA8::default();
            let lv = RGBA8::default();
            let mut val_r = blur_radius_next * (fv.r as isize);
            let mut val_g = blur_radius_next * (fv.g as isize);
            let mut val_b = blur_radius_next * (fv.b as isize);
            let mut val_a = blur_radius_next * (fv.a as isize);
            let get_left = |i: usize| {
                if i < row_start { fv } else { backbuf.data[i] }
            };
            let get_right = |i: usize| {
                if i > row_end { lv } else { backbuf.data[i] }
            };
            for j in 0..cmp::min(blur_radius, width) {
                let bb = backbuf.data[ti + j];
                val_r += bb.r as isize;
                val_g += bb.g as isize;
                val_b += bb.b as isize;
                val_a += bb.a as isize;
            }
            if blur_radius > width {
                val_r += blur_radius_prev * (lv.r as isize);
                val_g += blur_radius_prev * (lv.g as isize);
                val_b += blur_radius_prev * (lv.b as isize);
                val_a += blur_radius_prev * (lv.a as isize);
            }
            for _ in 0..cmp::min(width, blur_radius + 1) {
                let bb = get_right(ri);
                ri += 1;
                val_r += sub(bb.r, fv.r);
                val_g += sub(bb.g, fv.g);
                val_b += sub(bb.b, fv.b);
                val_a += sub(bb.a, fv.a);
                frontbuf.data[ti] = RGBA8 {
                    r: round(val_r as f32 * iarr) as u8,
                    g: round(val_g as f32 * iarr) as u8,
                    b: round(val_b as f32 * iarr) as u8,
                    a: round(val_a as f32 * iarr) as u8,
                };
                ti += 1;
            }
            if width <= blur_radius {
                continue;
            }
            for _ in (blur_radius + 1)..(width - blur_radius) {
                let bb1 = backbuf.data[ri];
                ri += 1;
                let bb2 = backbuf.data[li];
                li += 1;
                val_r += sub(bb1.r, bb2.r);
                val_g += sub(bb1.g, bb2.g);
                val_b += sub(bb1.b, bb2.b);
                val_a += sub(bb1.a, bb2.a);
                frontbuf.data[ti] = RGBA8 {
                    r: round(val_r as f32 * iarr) as u8,
                    g: round(val_g as f32 * iarr) as u8,
                    b: round(val_b as f32 * iarr) as u8,
                    a: round(val_a as f32 * iarr) as u8,
                };
                ti += 1;
            }
            for _ in 0..cmp::min(width - blur_radius - 1, blur_radius) {
                let bb = get_left(li);
                li += 1;
                val_r += sub(lv.r, bb.r);
                val_g += sub(lv.g, bb.g);
                val_b += sub(lv.b, bb.b);
                val_a += sub(lv.a, bb.a);
                frontbuf.data[ti] = RGBA8 {
                    r: round(val_r as f32 * iarr) as u8,
                    g: round(val_g as f32 * iarr) as u8,
                    b: round(val_b as f32 * iarr) as u8,
                    a: round(val_a as f32 * iarr) as u8,
                };
                ti += 1;
            }
        }
    }

    #[inline]
    fn round(mut x: f32) -> f32 {
        x += 12582912.0;
        x -= 12582912.0;
        x
    }
    #[inline]
    fn sub(c1: u8, c2: u8) -> isize {
        c1 as isize - c2 as isize
    }
}

// ===========================================================================
// Box blur OPTIMIZED (matching optimized box_blur.rs)
// ===========================================================================
mod box_blur_opt {
    use super::ImageRefMut;
    use rgb::RGBA8;
    use std::cmp;

    const STEPS: usize = 5;
    const VERT_TILE_W: usize = 16;

    pub fn apply(sigma_x: f64, sigma_y: f64, src: ImageRefMut) {
        let pixel_count = (src.width as usize) * (src.height as usize);

        // Quick threshold check matching production code (box_blur.rs):
        // sigma >= 10 gives max_radius >= 8 for STEPS=5 box blur.
        // Only use the tiled vertical path for large images with high sigma,
        // where the cache-locality benefit outweighs the overhead.
        if pixel_count > 1_000_000 && sigma_y >= 10.0 {
            apply_tiled(sigma_x, sigma_y, src);
        } else {
            // Below threshold: identical to naive (same code path in production).
            super::box_blur_naive::apply(sigma_x, sigma_y, src);
        }
    }

    fn apply_tiled(sigma_x: f64, sigma_y: f64, mut src: ImageRefMut) {
        let boxes_horz = create_box_gauss(sigma_x as f32);
        let boxes_vert = create_box_gauss(sigma_y as f32);
        let mut backbuf = src.data.to_vec();
        let mut backbuf = ImageRefMut::new(src.width, src.height, &mut backbuf);
        for (box_size_horz, box_size_vert) in boxes_horz.iter().zip(boxes_vert.iter()) {
            let radius_horz = ((box_size_horz - 1) / 2) as usize;
            let radius_vert = ((box_size_vert - 1) / 2) as usize;
            box_blur_impl(radius_horz, radius_vert, &mut backbuf, &mut src);
        }
    }

    fn create_box_gauss(sigma: f32) -> [i32; STEPS] {
        if sigma > 0.0 {
            let n_float = STEPS as f32;
            let w_ideal = (12.0 * sigma * sigma / n_float).sqrt() + 1.0;
            let mut wl = w_ideal.floor() as i32;
            if wl % 2 == 0 {
                wl -= 1;
            }
            let wu = wl + 2;
            let wl_float = wl as f32;
            let m_ideal = (12.0 * sigma * sigma
                - n_float * wl_float * wl_float
                - 4.0 * n_float * wl_float
                - 3.0 * n_float)
                / (-4.0 * wl_float - 4.0);
            let m = m_ideal.round() as usize;
            let mut sizes = [0; STEPS];
            for i in 0..STEPS {
                if i < m {
                    sizes[i] = wl;
                } else {
                    sizes[i] = wu;
                }
            }
            sizes
        } else {
            [1; STEPS]
        }
    }

    fn box_blur_impl(
        blur_radius_horz: usize,
        blur_radius_vert: usize,
        backbuf: &mut ImageRefMut,
        frontbuf: &mut ImageRefMut,
    ) {
        box_blur_vert_tiled(blur_radius_vert, frontbuf, backbuf);
        box_blur_horz_opt(blur_radius_horz, backbuf, frontbuf);
    }

    fn box_blur_vert_tiled(blur_radius: usize, backbuf: &ImageRefMut, frontbuf: &mut ImageRefMut) {
        if blur_radius == 0 {
            frontbuf.data.copy_from_slice(backbuf.data);
            return;
        }
        let width = backbuf.width as usize;
        let height = backbuf.height as usize;
        let iarr = 1.0 / (blur_radius + blur_radius + 1) as f32;
        let mut col = 0;
        while col < width {
            let tile_end = cmp::min(col + VERT_TILE_W, width);
            for i in col..tile_end {
                box_blur_vert_single_col(
                    blur_radius,
                    width,
                    height,
                    iarr,
                    i,
                    backbuf.data,
                    frontbuf.data,
                );
            }
            col = tile_end;
        }
    }

    #[inline]
    fn box_blur_vert_single_col(
        blur_radius: usize,
        width: usize,
        height: usize,
        iarr: f32,
        col: usize,
        backbuf: &[RGBA8],
        frontbuf: &mut [RGBA8],
    ) {
        let col_end = col + width * (height - 1);
        let mut ti = col;
        let mut li = col;
        let mut ri = col + blur_radius * width;
        let fv: [i32; 4] = [0; 4];
        let lv: [i32; 4] = [0; 4];
        let blur_radius_next = blur_radius as i32 + 1;
        let mut val: [i32; 4] = [
            blur_radius_next * fv[0],
            blur_radius_next * fv[1],
            blur_radius_next * fv[2],
            blur_radius_next * fv[3],
        ];

        #[inline(always)]
        fn px_to_arr(p: RGBA8) -> [i32; 4] {
            [p.r as i32, p.g as i32, p.b as i32, p.a as i32]
        }

        for j in 0..cmp::min(blur_radius, height) {
            let bb = px_to_arr(backbuf[ti + j * width]);
            val[0] += bb[0];
            val[1] += bb[1];
            val[2] += bb[2];
            val[3] += bb[3];
        }
        if blur_radius > height {
            let blur_radius_prev = blur_radius as i32 - height as i32;
            val[0] += blur_radius_prev * lv[0];
            val[1] += blur_radius_prev * lv[1];
            val[2] += blur_radius_prev * lv[2];
            val[3] += blur_radius_prev * lv[3];
        }
        for _ in 0..cmp::min(height, blur_radius + 1) {
            let bb = if ri > col_end {
                lv
            } else {
                px_to_arr(backbuf[ri])
            };
            ri += width;
            val[0] += bb[0] - fv[0];
            val[1] += bb[1] - fv[1];
            val[2] += bb[2] - fv[2];
            val[3] += bb[3] - fv[3];
            frontbuf[ti] = RGBA8 {
                r: round(val[0] as f32 * iarr) as u8,
                g: round(val[1] as f32 * iarr) as u8,
                b: round(val[2] as f32 * iarr) as u8,
                a: round(val[3] as f32 * iarr) as u8,
            };
            ti += width;
        }
        if height <= blur_radius {
            return;
        }
        for _ in (blur_radius + 1)..(height - blur_radius) {
            let bb1 = px_to_arr(backbuf[ri]);
            ri += width;
            let bb2 = px_to_arr(backbuf[li]);
            li += width;
            val[0] += bb1[0] - bb2[0];
            val[1] += bb1[1] - bb2[1];
            val[2] += bb1[2] - bb2[2];
            val[3] += bb1[3] - bb2[3];
            frontbuf[ti] = RGBA8 {
                r: round(val[0] as f32 * iarr) as u8,
                g: round(val[1] as f32 * iarr) as u8,
                b: round(val[2] as f32 * iarr) as u8,
                a: round(val[3] as f32 * iarr) as u8,
            };
            ti += width;
        }
        for _ in 0..cmp::min(height - blur_radius - 1, blur_radius) {
            let bb = if li < col { fv } else { px_to_arr(backbuf[li]) };
            li += width;
            val[0] += lv[0] - bb[0];
            val[1] += lv[1] - bb[1];
            val[2] += lv[2] - bb[2];
            val[3] += lv[3] - bb[3];
            frontbuf[ti] = RGBA8 {
                r: round(val[0] as f32 * iarr) as u8,
                g: round(val[1] as f32 * iarr) as u8,
                b: round(val[2] as f32 * iarr) as u8,
                a: round(val[3] as f32 * iarr) as u8,
            };
            ti += width;
        }
    }

    fn box_blur_horz_opt(blur_radius: usize, backbuf: &ImageRefMut, frontbuf: &mut ImageRefMut) {
        if blur_radius == 0 {
            frontbuf.data.copy_from_slice(backbuf.data);
            return;
        }
        let width = backbuf.width as usize;
        let height = backbuf.height as usize;
        let iarr = 1.0 / (blur_radius + blur_radius + 1) as f32;

        #[inline(always)]
        fn px_to_arr(p: RGBA8) -> [i32; 4] {
            [p.r as i32, p.g as i32, p.b as i32, p.a as i32]
        }

        for i in 0..height {
            let row_start = i * width;
            let row_end = (i + 1) * width - 1;
            let mut ti = row_start;
            let mut li = ti;
            let mut ri = ti + blur_radius;
            let fv: [i32; 4] = [0; 4];
            let lv: [i32; 4] = [0; 4];
            let blur_radius_next = blur_radius as i32 + 1;
            let mut val: [i32; 4] = [
                blur_radius_next * fv[0],
                blur_radius_next * fv[1],
                blur_radius_next * fv[2],
                blur_radius_next * fv[3],
            ];
            for j in 0..cmp::min(blur_radius, width) {
                let bb = px_to_arr(backbuf.data[ti + j]);
                val[0] += bb[0];
                val[1] += bb[1];
                val[2] += bb[2];
                val[3] += bb[3];
            }
            if blur_radius > width {
                let blur_radius_prev = blur_radius as i32 - width as i32;
                val[0] += blur_radius_prev * lv[0];
                val[1] += blur_radius_prev * lv[1];
                val[2] += blur_radius_prev * lv[2];
                val[3] += blur_radius_prev * lv[3];
            }
            for _ in 0..cmp::min(width, blur_radius + 1) {
                let bb = if ri > row_end {
                    lv
                } else {
                    px_to_arr(backbuf.data[ri])
                };
                ri += 1;
                val[0] += bb[0] - fv[0];
                val[1] += bb[1] - fv[1];
                val[2] += bb[2] - fv[2];
                val[3] += bb[3] - fv[3];
                frontbuf.data[ti] = RGBA8 {
                    r: round(val[0] as f32 * iarr) as u8,
                    g: round(val[1] as f32 * iarr) as u8,
                    b: round(val[2] as f32 * iarr) as u8,
                    a: round(val[3] as f32 * iarr) as u8,
                };
                ti += 1;
            }
            if width <= blur_radius {
                continue;
            }
            for _ in (blur_radius + 1)..(width - blur_radius) {
                let bb1 = px_to_arr(backbuf.data[ri]);
                ri += 1;
                let bb2 = px_to_arr(backbuf.data[li]);
                li += 1;
                val[0] += bb1[0] - bb2[0];
                val[1] += bb1[1] - bb2[1];
                val[2] += bb1[2] - bb2[2];
                val[3] += bb1[3] - bb2[3];
                frontbuf.data[ti] = RGBA8 {
                    r: round(val[0] as f32 * iarr) as u8,
                    g: round(val[1] as f32 * iarr) as u8,
                    b: round(val[2] as f32 * iarr) as u8,
                    a: round(val[3] as f32 * iarr) as u8,
                };
                ti += 1;
            }
            for _ in 0..cmp::min(width - blur_radius - 1, blur_radius) {
                let bb = if li < row_start {
                    fv
                } else {
                    px_to_arr(backbuf.data[li])
                };
                li += 1;
                val[0] += lv[0] - bb[0];
                val[1] += lv[1] - bb[1];
                val[2] += lv[2] - bb[2];
                val[3] += lv[3] - bb[3];
                frontbuf.data[ti] = RGBA8 {
                    r: round(val[0] as f32 * iarr) as u8,
                    g: round(val[1] as f32 * iarr) as u8,
                    b: round(val[2] as f32 * iarr) as u8,
                    a: round(val[3] as f32 * iarr) as u8,
                };
                ti += 1;
            }
        }
    }

    #[inline]
    fn round(mut x: f32) -> f32 {
        x += 12582912.0;
        x -= 12582912.0;
        x
    }
}

// ===========================================================================
// IIR blur NAIVE (verbatim copy from original iir_blur.rs)
// ===========================================================================
mod iir_blur_naive {
    use super::ImageRefMut;
    use rgb::ComponentSlice;

    struct BlurData {
        width: usize,
        height: usize,
        sigma_x: f64,
        sigma_y: f64,
        steps: usize,
    }

    pub fn apply(sigma_x: f64, sigma_y: f64, src: ImageRefMut) {
        let buf_size = (src.width * src.height) as usize;
        let mut buf = vec![0.0f64; buf_size];
        let d = BlurData {
            width: src.width as usize,
            height: src.height as usize,
            sigma_x,
            sigma_y,
            steps: 4,
        };
        let data = src.data.as_mut_slice();
        gaussian_channel(data, &d, 0, &mut buf);
        gaussian_channel(data, &d, 1, &mut buf);
        gaussian_channel(data, &d, 2, &mut buf);
        gaussian_channel(data, &d, 3, &mut buf);
    }

    fn gaussian_channel(data: &mut [u8], d: &BlurData, channel: usize, buf: &mut [f64]) {
        for i in 0..data.len() / 4 {
            buf[i] = data[i * 4 + channel] as f64 / 255.0;
        }
        gaussianiir2d(d, buf);
        for i in 0..data.len() / 4 {
            data[i * 4 + channel] = (buf[i] * 255.0) as u8;
        }
    }

    fn gaussianiir2d(d: &BlurData, buf: &mut [f64]) {
        let (lambda_x, dnu_x) = if d.sigma_x > 0.0 {
            let (lambda, dnu) = gen_coefficients(d.sigma_x, d.steps);
            for y in 0..d.height {
                for _ in 0..d.steps {
                    let idx = d.width * y;
                    for x in 1..d.width {
                        buf[idx + x] += dnu * buf[idx + x - 1];
                    }
                    let mut x = d.width - 1;
                    while x > 0 {
                        buf[idx + x - 1] += dnu * buf[idx + x];
                        x -= 1;
                    }
                }
            }
            (lambda, dnu)
        } else {
            (1.0, 1.0)
        };
        let (lambda_y, dnu_y) = if d.sigma_y > 0.0 {
            let (lambda, dnu) = gen_coefficients(d.sigma_y, d.steps);
            for x in 0..d.width {
                for _ in 0..d.steps {
                    let idx = x;
                    let mut y = d.width;
                    while y < buf.len() {
                        buf[idx + y] += dnu * buf[idx + y - d.width];
                        y += d.width;
                    }
                    y = buf.len() - d.width;
                    while y > 0 {
                        buf[idx + y - d.width] += dnu * buf[idx + y];
                        y -= d.width;
                    }
                }
            }
            (lambda, dnu)
        } else {
            (1.0, 1.0)
        };
        let post_scale =
            ((dnu_x * dnu_y).sqrt() / (lambda_x * lambda_y).sqrt()).powi(2 * d.steps as i32);
        buf.iter_mut().for_each(|v| *v *= post_scale);
    }

    fn gen_coefficients(sigma: f64, steps: usize) -> (f64, f64) {
        let lambda = (sigma * sigma) / (2.0 * steps as f64);
        let dnu = (1.0 + 2.0 * lambda - (1.0 + 4.0 * lambda).sqrt()) / (2.0 * lambda);
        (lambda, dnu)
    }
}

// ===========================================================================
// IIR blur OPTIMIZED (matching optimized iir_blur.rs)
// ===========================================================================
mod iir_blur_opt {
    use super::ImageRefMut;
    use rgb::ComponentSlice;
    use std::cmp;

    const IIR_VERT_TILE_W: usize = 32;

    pub fn apply(sigma_x: f64, sigma_y: f64, src: ImageRefMut) {
        let width = src.width as usize;
        let height = src.height as usize;
        let pixel_count = width * height;
        let steps = 4usize;

        let mut buf = vec![[0.0f64; 4]; pixel_count];
        let data = src.data.as_mut_slice();

        for i in 0..pixel_count {
            let base = i * 4;
            buf[i] = [
                data[base] as f64 / 255.0,
                data[base + 1] as f64 / 255.0,
                data[base + 2] as f64 / 255.0,
                data[base + 3] as f64 / 255.0,
            ];
        }

        let (lambda_x, dnu_x) = if sigma_x > 0.0 {
            let (lambda, dnu) = gen_coefficients(sigma_x, steps);
            for y in 0..height {
                let idx = width * y;
                for _ in 0..steps {
                    for x in 1..width {
                        let prev = buf[idx + x - 1];
                        let cur = &mut buf[idx + x];
                        cur[0] += dnu * prev[0];
                        cur[1] += dnu * prev[1];
                        cur[2] += dnu * prev[2];
                        cur[3] += dnu * prev[3];
                    }
                    let mut x = width - 1;
                    while x > 0 {
                        let next = buf[idx + x];
                        let cur = &mut buf[idx + x - 1];
                        cur[0] += dnu * next[0];
                        cur[1] += dnu * next[1];
                        cur[2] += dnu * next[2];
                        cur[3] += dnu * next[3];
                        x -= 1;
                    }
                }
            }
            (lambda, dnu)
        } else {
            (1.0, 1.0)
        };

        let (lambda_y, dnu_y) = if sigma_y > 0.0 {
            let (lambda, dnu) = gen_coefficients(sigma_y, steps);
            let mut col = 0;
            while col < width {
                let tile_end = cmp::min(col + IIR_VERT_TILE_W, width);
                for x in col..tile_end {
                    for _ in 0..steps {
                        let mut y_off = width;
                        while y_off < buf.len() {
                            let prev = buf[x + y_off - width];
                            let cur = &mut buf[x + y_off];
                            cur[0] += dnu * prev[0];
                            cur[1] += dnu * prev[1];
                            cur[2] += dnu * prev[2];
                            cur[3] += dnu * prev[3];
                            y_off += width;
                        }
                        y_off = buf.len() - width;
                        while y_off > 0 {
                            let next = buf[x + y_off];
                            let cur = &mut buf[x + y_off - width];
                            cur[0] += dnu * next[0];
                            cur[1] += dnu * next[1];
                            cur[2] += dnu * next[2];
                            cur[3] += dnu * next[3];
                            y_off -= width;
                        }
                    }
                }
                col = tile_end;
            }
            (lambda, dnu)
        } else {
            (1.0, 1.0)
        };

        let post_scale =
            ((dnu_x * dnu_y).sqrt() / (lambda_x * lambda_y).sqrt()).powi(2 * steps as i32);

        for i in 0..pixel_count {
            let base = i * 4;
            let px = buf[i];
            data[base] = (px[0] * post_scale * 255.0) as u8;
            data[base + 1] = (px[1] * post_scale * 255.0) as u8;
            data[base + 2] = (px[2] * post_scale * 255.0) as u8;
            data[base + 3] = (px[3] * post_scale * 255.0) as u8;
        }
    }

    fn gen_coefficients(sigma: f64, steps: usize) -> (f64, f64) {
        let lambda = (sigma * sigma) / (2.0 * steps as f64);
        let dnu = (1.0 + 2.0 * lambda - (1.0 + 4.0 * lambda).sqrt()) / (2.0 * lambda);
        (lambda, dnu)
    }
}

// ===========================================================================
// Benchmark harness
// ===========================================================================

fn make_test_image(width: u32, height: u32) -> Vec<RGBA8> {
    let n = (width * height) as usize;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(RGBA8 {
            r: (i % 256) as u8,
            g: ((i * 7) % 256) as u8,
            b: ((i * 13) % 256) as u8,
            a: ((i * 3 + 128) % 256) as u8,
        });
    }
    data
}

fn bench_one(
    label: &str,
    width: u32,
    height: u32,
    sigma: f64,
    f: &dyn Fn(f64, f64, ImageRefMut),
    iters: u32,
) -> f64 {
    let original = make_test_image(width, height);

    // warmup
    for _ in 0..2 {
        let mut data = original.clone();
        f(sigma, sigma, ImageRefMut::new(width, height, &mut data));
    }

    let start = Instant::now();
    for _ in 0..iters {
        let mut data = original.clone();
        f(sigma, sigma, ImageRefMut::new(width, height, &mut data));
    }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_micros() as f64 / iters as f64;
    let mpix_per_sec = (width as f64 * height as f64) / per_iter_us;

    println!(
        "{:<60} {:>10.1} us/iter  ({:.1} Mpix/s)",
        label, per_iter_us, mpix_per_sec
    );
    per_iter_us
}

fn choose_iters(width: u32, height: u32) -> u32 {
    let pixels = (width as u64) * (height as u64);
    if pixels > 500_000 {
        5
    } else if pixels > 100_000 {
        15
    } else if pixels > 10_000 {
        50
    } else {
        200
    }
}

fn main() {
    println!("=== feGaussianBlur Real-World Benchmark ===\n");

    // -----------------------------------------------------------------
    // 1. IIR blur path: sigma < 2 (subtle glow, anti-alias)
    // -----------------------------------------------------------------
    println!("--- IIR blur (sigma < 2) ---");
    println!("  Use case: subtle anti-alias, glow, fine detail softening\n");

    let iir_configs: &[(u32, u32, &str)] = &[
        (48, 48, "large icon"),
        (96, 96, "avatar"),
        (400, 300, "card"),
        (800, 600, "tablet"),
    ];
    let iir_sigma = 1.5;

    for &(w, h, label) in iir_configs {
        let iters = choose_iters(w, h);
        let tag_naive = format!("IIR {}x{} ({}) sigma={} naive", w, h, label, iir_sigma);
        let tag_opt = format!("IIR {}x{} ({}) sigma={} optimized", w, h, label, iir_sigma);
        let t_naive = bench_one(&tag_naive, w, h, iir_sigma, &iir_blur_naive::apply, iters);
        let t_opt = bench_one(&tag_opt, w, h, iir_sigma, &iir_blur_opt::apply, iters);
        println!("  => speedup: {:.2}x\n", t_naive / t_opt);
    }

    // -----------------------------------------------------------------
    // 2. Box blur: icon shadows (sigma=2, sigma=4)
    // -----------------------------------------------------------------
    println!("--- Box blur: icon shadows (Material Z1-Z4) ---\n");

    let icon_sizes: &[(u32, u32, &str)] =
        &[(24, 24, "icon"), (48, 48, "large icon"), (96, 96, "avatar")];
    let icon_sigmas: &[(f64, &str)] = &[(2.0, "Material Z1-Z2"), (4.0, "MDN canonical shadow")];

    for &(w, h, size_label) in icon_sizes {
        for &(sigma, sigma_label) in icon_sigmas {
            let iters = choose_iters(w, h);
            let tag_naive = format!(
                "box {}x{} ({}) sigma={} ({}) naive",
                w, h, size_label, sigma, sigma_label
            );
            let tag_opt = format!(
                "box {}x{} ({}) sigma={} ({}) optimized",
                w, h, size_label, sigma, sigma_label
            );
            let t_naive = bench_one(&tag_naive, w, h, sigma, &box_blur_naive::apply, iters);
            let t_opt = bench_one(&tag_opt, w, h, sigma, &box_blur_opt::apply, iters);
            println!("  => speedup: {:.2}x\n", t_naive / t_opt);
        }
    }

    // -----------------------------------------------------------------
    // 3. Box blur: card/UI shadows (sigma=4, sigma=8)
    // -----------------------------------------------------------------
    println!("--- Box blur: card/UI shadows ---\n");

    let card_sizes: &[(u32, u32, &str)] = &[
        (200, 150, "thumbnail"),
        (400, 300, "card"),
        (800, 600, "tablet"),
    ];
    let card_sigmas: &[(f64, &str)] = &[(4.0, "standard shadow"), (8.0, "backdrop-blur default")];

    for &(w, h, size_label) in card_sizes {
        for &(sigma, sigma_label) in card_sigmas {
            let iters = choose_iters(w, h);
            let tag_naive = format!(
                "box {}x{} ({}) sigma={} ({}) naive",
                w, h, size_label, sigma, sigma_label
            );
            let tag_opt = format!(
                "box {}x{} ({}) sigma={} ({}) optimized",
                w, h, size_label, sigma, sigma_label
            );
            let t_naive = bench_one(&tag_naive, w, h, sigma, &box_blur_naive::apply, iters);
            let t_opt = bench_one(&tag_opt, w, h, sigma, &box_blur_opt::apply, iters);
            println!("  => speedup: {:.2}x\n", t_naive / t_opt);
        }
    }

    // -----------------------------------------------------------------
    // 4. Box blur: backdrop/frosted glass (sigma=16, sigma=40)
    // -----------------------------------------------------------------
    println!("--- Box blur: backdrop / frosted glass ---\n");

    let backdrop_sizes: &[(u32, u32, &str)] = &[
        (800, 600, "tablet"),
        (1024, 768, "laptop"),
        (1500, 1000, "desktop"),
    ];
    let backdrop_sigmas: &[(f64, &str)] =
        &[(16.0, "Apple frosted glass"), (40.0, "backdrop-blur-2xl")];

    for &(w, h, size_label) in backdrop_sizes {
        for &(sigma, sigma_label) in backdrop_sigmas {
            let iters = choose_iters(w, h);
            let tag_naive = format!(
                "box {}x{} ({}) sigma={} ({}) naive",
                w, h, size_label, sigma, sigma_label
            );
            let tag_opt = format!(
                "box {}x{} ({}) sigma={} ({}) optimized",
                w, h, size_label, sigma, sigma_label
            );
            let t_naive = bench_one(&tag_naive, w, h, sigma, &box_blur_naive::apply, iters);
            let t_opt = bench_one(&tag_opt, w, h, sigma, &box_blur_opt::apply, iters);
            println!("  => speedup: {:.2}x\n", t_naive / t_opt);
        }
    }

    // -----------------------------------------------------------------
    // 5. IIR interleaved: sigma < 2 on large images (>500k pixels)
    // -----------------------------------------------------------------
    println!("--- IIR interleaved: large image + small sigma ---\n");

    let iir_large_configs: &[(u32, u32, &str)] = &[(1024, 768, "laptop"), (1500, 1000, "desktop")];

    for &(w, h, label) in iir_large_configs {
        let iters = choose_iters(w, h);
        let tag_naive = format!("IIR {}x{} ({}) sigma={} naive", w, h, label, iir_sigma);
        let tag_opt = format!("IIR {}x{} ({}) sigma={} optimized", w, h, label, iir_sigma);
        let t_naive = bench_one(&tag_naive, w, h, iir_sigma, &iir_blur_naive::apply, iters);
        let t_opt = bench_one(&tag_opt, w, h, iir_sigma, &iir_blur_opt::apply, iters);
        println!("  => speedup: {:.2}x\n", t_naive / t_opt);
    }

    // -----------------------------------------------------------------
    // 6. Threshold boundary tests (critical for regression detection)
    // -----------------------------------------------------------------
    println!("--- Threshold boundary: 250k pixel box blur boundary ---");
    println!("  Tests: 499x500=249500, 500x500=250000, 501x500=250500\n");

    let box_boundary_sizes: &[(u32, u32, &str)] = &[
        (499, 500, "just below 250k"),
        (500, 500, "at 250k"),
        (501, 500, "just above 250k"),
    ];

    for &(w, h, label) in box_boundary_sizes {
        let iters = choose_iters(w, h);
        let sigma = 8.0;
        let tag_naive = format!(
            "box {}x{} ({}px, {}) sigma={} naive",
            w,
            h,
            w as u64 * h as u64,
            label,
            sigma
        );
        let tag_opt = format!(
            "box {}x{} ({}px, {}) sigma={} optimized",
            w,
            h,
            w as u64 * h as u64,
            label,
            sigma
        );
        let t_naive = bench_one(&tag_naive, w, h, sigma, &box_blur_naive::apply, iters);
        let t_opt = bench_one(&tag_opt, w, h, sigma, &box_blur_opt::apply, iters);
        println!("  => speedup: {:.2}x\n", t_naive / t_opt);
    }

    println!("--- Threshold boundary: 500k pixel IIR interleaved boundary ---");
    println!("  Tests: 707x707=499849, 708x707=500556\n");

    let iir_boundary_sizes: &[(u32, u32, &str)] =
        &[(707, 707, "just below 500k"), (708, 707, "just above 500k")];

    for &(w, h, label) in iir_boundary_sizes {
        let iters = choose_iters(w, h);
        let sigma = 1.5;
        let tag_naive = format!(
            "IIR {}x{} ({}px, {}) sigma={} naive",
            w,
            h,
            w as u64 * h as u64,
            label,
            sigma
        );
        let tag_opt = format!(
            "IIR {}x{} ({}px, {}) sigma={} optimized",
            w,
            h,
            w as u64 * h as u64,
            label,
            sigma
        );
        let t_naive = bench_one(&tag_naive, w, h, sigma, &iir_blur_naive::apply, iters);
        let t_opt = bench_one(&tag_opt, w, h, sigma, &iir_blur_opt::apply, iters);
        println!("  => speedup: {:.2}x\n", t_naive / t_opt);
    }
}
