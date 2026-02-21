// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Comprehensive performance regression benchmark for feGaussianBlur.
//!
//! Tests both box blur (sigma >= 2.0) and IIR blur (sigma < 2.0) across
//! a wide range of image sizes, sigma values, input patterns, and
//! asymmetric sigma configurations. Also tests around the 16x16 threshold
//! where the code switches between naive and optimized paths.
//!
//! Uses multithreading (std::thread::scope) to run independent benchmark
//! configurations in parallel across all available CPU cores.
//!
//! Run with: cargo run -p resvg --release --example bench_blur_comprehensive

#![allow(clippy::needless_range_loop)]

use rgb::RGBA8;
use std::sync::atomic::{AtomicUsize, Ordering};
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

    fn box_blur_inner(
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
                if i < col_start {
                    fv
                } else {
                    backbuf.data[i]
                }
            };
            let get_bottom = |i: usize| {
                if i > col_end {
                    lv
                } else {
                    backbuf.data[i]
                }
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
                if i < row_start {
                    fv
                } else {
                    backbuf.data[i]
                }
            };
            let get_right = |i: usize| {
                if i > row_end {
                    lv
                } else {
                    backbuf.data[i]
                }
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

    pub fn apply(sigma_x: f64, sigma_y: f64, mut src: ImageRefMut) {
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
// Input pattern generators
// ===========================================================================

fn make_opaque(width: u32, height: u32) -> Vec<RGBA8> {
    let n = (width * height) as usize;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(RGBA8 {
            r: (i % 256) as u8,
            g: ((i * 7) % 256) as u8,
            b: ((i * 13) % 256) as u8,
            a: 255,
        });
    }
    data
}

fn make_gradient_alpha(width: u32, height: u32) -> Vec<RGBA8> {
    let n = (width * height) as usize;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let alpha = ((i as f64 / n as f64) * 255.0) as u8;
        data.push(RGBA8 {
            r: (i % 256) as u8,
            g: ((i * 7) % 256) as u8,
            b: ((i * 13) % 256) as u8,
            a: alpha,
        });
    }
    data
}

fn make_random_alpha(width: u32, height: u32) -> Vec<RGBA8> {
    let n = (width * height) as usize;
    let mut data = Vec::with_capacity(n);
    // Simple LCG for deterministic "random" data
    let mut rng: u64 = 0xDEADBEEF;
    for _ in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = (rng >> 56) as u8;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let g = (rng >> 56) as u8;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let b = (rng >> 56) as u8;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a = (rng >> 56) as u8;
        data.push(RGBA8 { r, g, b, a });
    }
    data
}

// ===========================================================================
// Benchmark harness
// ===========================================================================

/// Which blur algorithm pair to benchmark (naive vs optimized).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    Box,
    Iir,
}

/// Which input pattern to generate.
#[derive(Clone, Copy, PartialEq, Eq)]
enum InputPattern {
    Opaque,
    Gradient,
    Random,
}

impl InputPattern {
    fn name(self) -> &'static str {
        match self {
            InputPattern::Opaque => "opaque",
            InputPattern::Gradient => "gradient",
            InputPattern::Random => "random",
        }
    }

    fn generate(self, width: u32, height: u32) -> Vec<RGBA8> {
        match self {
            InputPattern::Opaque => make_opaque(width, height),
            InputPattern::Gradient => make_gradient_alpha(width, height),
            InputPattern::Random => make_random_alpha(width, height),
        }
    }
}

/// Which section of the output this result belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    BoxBlur,
    IirBlur,
    AsymmetricSigma,
    ThresholdBoundary,
}

/// A single benchmark configuration to run.
struct Config {
    order: usize,
    section: Section,
    width: u32,
    height: u32,
    sigma_x: f64,
    sigma_y: f64,
    pattern: InputPattern,
    algorithm: Algorithm,
    /// Display label for the algorithm column.
    algorithm_label: String,
}

struct BenchResult {
    order: usize,
    section: Section,
    image_size: String,
    sigma: String,
    input_pattern: String,
    algorithm: String,
    naive_us: f64,
    opt_us: f64,
    speedup: f64,
}

fn bench_one_fn(
    data_template: &[RGBA8],
    width: u32,
    height: u32,
    sigma_x: f64,
    sigma_y: f64,
    f: &dyn Fn(f64, f64, ImageRefMut),
    iters: u32,
) -> f64 {
    // warmup
    for _ in 0..2 {
        let mut data = data_template.to_vec();
        f(sigma_x, sigma_y, ImageRefMut::new(width, height, &mut data));
    }

    let start = Instant::now();
    for _ in 0..iters {
        let mut data = data_template.to_vec();
        f(sigma_x, sigma_y, ImageRefMut::new(width, height, &mut data));
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / iters as f64 / 1000.0 // microseconds
}

fn choose_iters(width: u32, height: u32) -> u32 {
    let pixels = (width as u64) * (height as u64);
    if pixels > 500_000 {
        5
    } else if pixels > 50_000 {
        20
    } else if pixels > 5_000 {
        100
    } else {
        500
    }
}

/// Run a single benchmark configuration and return the result.
fn run_config(config: &Config, progress: &AtomicUsize, total: usize) -> BenchResult {
    let data_template = config.pattern.generate(config.width, config.height);
    let iters = choose_iters(config.width, config.height);

    let (naive_fn, opt_fn): (
        &dyn Fn(f64, f64, ImageRefMut),
        &dyn Fn(f64, f64, ImageRefMut),
    ) = match config.algorithm {
        Algorithm::Box => (&box_blur_naive::apply, &box_blur_opt::apply),
        Algorithm::Iir => (&iir_blur_naive::apply, &iir_blur_opt::apply),
    };

    let t_naive = bench_one_fn(
        &data_template,
        config.width,
        config.height,
        config.sigma_x,
        config.sigma_y,
        naive_fn,
        iters,
    );
    let t_opt = bench_one_fn(
        &data_template,
        config.width,
        config.height,
        config.sigma_x,
        config.sigma_y,
        opt_fn,
        iters,
    );
    let speedup = t_naive / t_opt;

    let size_str = format!("{}x{}", config.width, config.height);
    let sigma_str = if config.sigma_x == config.sigma_y {
        format!("{}", config.sigma_x)
    } else {
        format!("({},{})", config.sigma_x, config.sigma_y)
    };

    let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
    eprint!("\r  Progress: {}/{} configurations completed...", done, total);

    BenchResult {
        order: config.order,
        section: config.section,
        image_size: size_str,
        sigma: sigma_str,
        input_pattern: config.pattern.name().to_string(),
        algorithm: config.algorithm_label.clone(),
        naive_us: t_naive,
        opt_us: t_opt,
        speedup,
    }
}

fn main() {
    // =====================================================================
    // Part 1: Correctness verification (sequential -- fast, has dependencies)
    // =====================================================================
    println!("=== Correctness Verification ===\n");

    let correctness_sizes: &[(u32, u32)] = &[
        (4, 4),
        (8, 8),
        (15, 15),
        (16, 16),
        (17, 17),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (15, 64),
        (64, 15),
    ];

    let mut all_correct = true;

    // Box blur correctness
    println!("--- Box Blur Correctness ---");
    for &(w, h) in correctness_sizes {
        for &sigma in &[2.0, 5.0, 10.0, 50.0] {
            let original = make_random_alpha(w, h);
            let mut data_naive = original.clone();
            box_blur_naive::apply(sigma, sigma, ImageRefMut::new(w, h, &mut data_naive));
            let mut data_opt = original.clone();
            box_blur_opt::apply(sigma, sigma, ImageRefMut::new(w, h, &mut data_opt));
            if data_naive == data_opt {
                println!("  PASS: {}x{} sigma={}", w, h, sigma);
            } else {
                let diff_count = data_naive
                    .iter()
                    .zip(data_opt.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                println!(
                    "  FAIL: {}x{} sigma={} ({} pixels differ)",
                    w, h, sigma, diff_count
                );
                all_correct = false;
            }
        }
    }

    // IIR blur correctness
    println!("\n--- IIR Blur Correctness ---");
    for &(w, h) in correctness_sizes {
        for &sigma in &[0.3, 0.5, 1.0, 1.5, 1.9] {
            let original = make_random_alpha(w, h);
            let mut data_naive = original.clone();
            iir_blur_naive::apply(sigma, sigma, ImageRefMut::new(w, h, &mut data_naive));
            let mut data_opt = original.clone();
            iir_blur_opt::apply(sigma, sigma, ImageRefMut::new(w, h, &mut data_opt));
            if data_naive == data_opt {
                println!("  PASS: {}x{} sigma={}", w, h, sigma);
            } else {
                let diff_count = data_naive
                    .iter()
                    .zip(data_opt.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                println!(
                    "  FAIL: {}x{} sigma={} ({} pixels differ)",
                    w, h, sigma, diff_count
                );
                all_correct = false;
            }
        }
    }

    // Asymmetric sigma correctness (box blur)
    println!("\n--- Asymmetric Sigma Correctness (Box Blur) ---");
    for &(w, h) in &[(64u32, 64u32), (128, 128)] {
        for &(sx, sy) in &[(5.0, 0.0), (0.0, 5.0), (3.0, 10.0)] {
            let original = make_random_alpha(w, h);
            let mut data_naive = original.clone();
            box_blur_naive::apply(sx, sy, ImageRefMut::new(w, h, &mut data_naive));
            let mut data_opt = original.clone();
            box_blur_opt::apply(sx, sy, ImageRefMut::new(w, h, &mut data_opt));
            if data_naive == data_opt {
                println!("  PASS: {}x{} sigma=({},{})", w, h, sx, sy);
            } else {
                let diff_count = data_naive
                    .iter()
                    .zip(data_opt.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                println!(
                    "  FAIL: {}x{} sigma=({},{}) ({} pixels differ)",
                    w, h, sx, sy, diff_count
                );
                all_correct = false;
            }
        }
    }

    if !all_correct {
        println!("\nWARNING: Some correctness tests FAILED!");
    } else {
        println!("\nAll correctness tests PASSED.");
    }

    // =====================================================================
    // Part 2: Comprehensive performance benchmark (parallel)
    // =====================================================================
    println!("\n=== Comprehensive Performance Benchmark ===\n");

    let image_sizes: &[(u32, u32)] = &[
        (4, 4),
        (8, 8),
        (15, 15),
        (16, 16),
        (17, 17),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ];

    let box_sigmas: &[f64] = &[2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0];
    let iir_sigmas: &[f64] = &[0.3, 0.5, 1.0, 1.5, 1.9];

    let input_patterns: &[InputPattern] = &[
        InputPattern::Opaque,
        InputPattern::Gradient,
        InputPattern::Random,
    ];

    let asym_sizes: &[(u32, u32)] = &[(64, 64), (256, 256), (512, 512)];
    let asym_sigmas: &[(f64, f64)] = &[(5.0, 0.0), (0.0, 5.0), (3.0, 10.0)];

    let threshold_sizes: &[(u32, u32)] = &[
        (15, 64),
        (15, 256),
        (64, 15),
        (256, 15),
        (16, 16),
        (15, 15),
    ];
    let threshold_sigmas_box: &[f64] = &[3.0, 10.0];
    let threshold_sigmas_iir: &[f64] = &[0.5, 1.5];

    // -----------------------------------------------------------------
    // Build all configurations upfront
    // -----------------------------------------------------------------
    let mut configs: Vec<Config> = Vec::new();
    let mut order: usize = 0;

    // Section 1: Box Blur Benchmark
    for &(w, h) in image_sizes {
        for &sigma in box_sigmas {
            for &pat in input_patterns {
                configs.push(Config {
                    order,
                    section: Section::BoxBlur,
                    width: w,
                    height: h,
                    sigma_x: sigma,
                    sigma_y: sigma,
                    pattern: pat,
                    algorithm: Algorithm::Box,
                    algorithm_label: "box".to_string(),
                });
                order += 1;
            }
        }
    }

    // Section 2: IIR Blur Benchmark
    for &(w, h) in image_sizes {
        for &sigma in iir_sigmas {
            for &pat in input_patterns {
                configs.push(Config {
                    order,
                    section: Section::IirBlur,
                    width: w,
                    height: h,
                    sigma_x: sigma,
                    sigma_y: sigma,
                    pattern: pat,
                    algorithm: Algorithm::Iir,
                    algorithm_label: "IIR".to_string(),
                });
                order += 1;
            }
        }
    }

    // Section 3: Asymmetric Sigma Benchmark (Box Blur)
    for &(w, h) in asym_sizes {
        for &(sx, sy) in asym_sigmas {
            for &pat in input_patterns {
                configs.push(Config {
                    order,
                    section: Section::AsymmetricSigma,
                    width: w,
                    height: h,
                    sigma_x: sx,
                    sigma_y: sy,
                    pattern: pat,
                    algorithm: Algorithm::Box,
                    algorithm_label: "box-asym".to_string(),
                });
                order += 1;
            }
        }
    }

    // Section 4: Threshold Boundary Tests
    for &(w, h) in threshold_sizes {
        // Box blur threshold tests
        for &sigma in threshold_sigmas_box {
            configs.push(Config {
                order,
                section: Section::ThresholdBoundary,
                width: w,
                height: h,
                sigma_x: sigma,
                sigma_y: sigma,
                pattern: InputPattern::Random,
                algorithm: Algorithm::Box,
                algorithm_label: "box".to_string(),
            });
            order += 1;
        }
        // IIR blur threshold tests
        for &sigma in threshold_sigmas_iir {
            configs.push(Config {
                order,
                section: Section::ThresholdBoundary,
                width: w,
                height: h,
                sigma_x: sigma,
                sigma_y: sigma,
                pattern: InputPattern::Random,
                algorithm: Algorithm::Iir,
                algorithm_label: "IIR".to_string(),
            });
            order += 1;
        }
    }

    let total = configs.len();
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!(
        "Running {} benchmark configurations across {} threads...\n",
        total, num_threads
    );

    // -----------------------------------------------------------------
    // Parallel execution using std::thread::scope
    // -----------------------------------------------------------------
    let progress = AtomicUsize::new(0);
    let chunk_size = (total + num_threads - 1) / num_threads;
    let config_chunks: Vec<&[Config]> = configs.chunks(chunk_size).collect();

    let mut all_results: Vec<BenchResult> = std::thread::scope(|s| {
        let handles: Vec<_> = config_chunks
            .into_iter()
            .map(|chunk| {
                let progress_ref = &progress;
                s.spawn(move || {
                    let mut results = Vec::with_capacity(chunk.len());
                    for config in chunk {
                        results.push(run_config(config, progress_ref, total));
                    }
                    results
                })
            })
            .collect();

        let mut merged = Vec::with_capacity(total);
        for handle in handles {
            merged.extend(handle.join().unwrap());
        }
        merged
    });

    // Clear progress line
    eprintln!();

    // Sort by original order to restore deterministic output
    all_results.sort_by_key(|r| r.order);

    // -----------------------------------------------------------------
    // Display results by section (same formatting as before)
    // -----------------------------------------------------------------

    // Section 1: Box Blur Benchmark
    println!("--- Box Blur Benchmark ---");
    println!(
        "{:<12} {:<8} {:<10} {:<10} {:>12} {:>12} {:>10}",
        "Image Size", "Sigma", "Input", "Algorithm", "Naive (us)", "Opt (us)", "Speedup"
    );
    println!("{}", "-".repeat(80));

    let mut box_results: Vec<&BenchResult> = Vec::new();
    let mut iir_results: Vec<&BenchResult> = Vec::new();

    for r in &all_results {
        if r.section == Section::BoxBlur {
            let regression_marker = if r.speedup < 0.95 { " *** REGRESSION" } else { "" };
            println!(
                "{:<12} {:<8} {:<10} {:<10} {:>12.1} {:>12.1} {:>9.2}x{}",
                r.image_size, r.sigma, r.input_pattern, r.algorithm, r.naive_us, r.opt_us,
                r.speedup, regression_marker
            );
            box_results.push(r);
        }
    }

    // Section 2: IIR Blur Benchmark
    println!("\n--- IIR Blur Benchmark ---");
    println!(
        "{:<12} {:<8} {:<10} {:<10} {:>12} {:>12} {:>10}",
        "Image Size", "Sigma", "Input", "Algorithm", "Naive (us)", "Opt (us)", "Speedup"
    );
    println!("{}", "-".repeat(80));

    for r in &all_results {
        if r.section == Section::IirBlur {
            let regression_marker = if r.speedup < 0.95 { " *** REGRESSION" } else { "" };
            println!(
                "{:<12} {:<8} {:<10} {:<10} {:>12.1} {:>12.1} {:>9.2}x{}",
                r.image_size, r.sigma, r.input_pattern, r.algorithm, r.naive_us, r.opt_us,
                r.speedup, regression_marker
            );
            iir_results.push(r);
        }
    }

    // Section 3: Asymmetric Sigma Benchmark (Box Blur)
    println!("\n--- Asymmetric Sigma Benchmark (Box Blur) ---");
    println!(
        "{:<12} {:<12} {:<10} {:<10} {:>12} {:>12} {:>10}",
        "Image Size", "Sigma", "Input", "Algorithm", "Naive (us)", "Opt (us)", "Speedup"
    );
    println!("{}", "-".repeat(84));

    for r in &all_results {
        if r.section == Section::AsymmetricSigma {
            let regression_marker = if r.speedup < 0.95 { " *** REGRESSION" } else { "" };
            println!(
                "{:<12} {:<12} {:<10} {:<10} {:>12.1} {:>12.1} {:>9.2}x{}",
                r.image_size, r.sigma, r.input_pattern, r.algorithm, r.naive_us, r.opt_us,
                r.speedup, regression_marker
            );
            box_results.push(r);
        }
    }

    // Section 4: Threshold Boundary Tests
    println!("\n--- Threshold Boundary Tests (w or h < 16 vs >= 16) ---");
    println!(
        "{:<12} {:<8} {:<10} {:<10} {:>12} {:>12} {:>10}",
        "Image Size", "Sigma", "Input", "Algorithm", "Naive (us)", "Opt (us)", "Speedup"
    );
    println!("{}", "-".repeat(80));

    for r in &all_results {
        if r.section == Section::ThresholdBoundary {
            // Parse width and height from image_size to determine the note
            let parts: Vec<&str> = r.image_size.split('x').collect();
            let w: u32 = parts[0].parse().unwrap_or(0);
            let h: u32 = parts[1].parse().unwrap_or(0);
            let note = if w < 16 || h < 16 {
                " (should use naive)"
            } else {
                " (should use opt)"
            };
            let regression_marker = if r.speedup < 0.95 { " *** REGRESSION" } else { "" };
            println!(
                "{:<12} {:<8} {:<10} {:<10} {:>12.1} {:>12.1} {:>9.2}x{}{}",
                r.image_size, r.sigma, r.input_pattern, r.algorithm, r.naive_us, r.opt_us,
                r.speedup, regression_marker, note
            );
        }
    }

    // =====================================================================
    // Part 3: Summary - flag regressions
    // =====================================================================
    println!("\n=== Regression Summary ===\n");

    let summary_results: Vec<&BenchResult> = box_results
        .iter()
        .copied()
        .chain(iir_results.iter().copied())
        .collect();

    let regressions: Vec<&&BenchResult> = summary_results
        .iter()
        .filter(|r| r.speedup < 0.95)
        .collect();

    if regressions.is_empty() {
        println!("No regressions detected (all speedups >= 0.95x).");
    } else {
        println!(
            "FOUND {} REGRESSIONS (speedup < 0.95x):\n",
            regressions.len()
        );
        println!(
            "{:<12} {:<12} {:<10} {:<10} {:>12} {:>12} {:>10}",
            "Image Size", "Sigma", "Input", "Algorithm", "Naive (us)", "Opt (us)", "Speedup"
        );
        println!("{}", "-".repeat(84));
        for r in &regressions {
            println!(
                "{:<12} {:<12} {:<10} {:<10} {:>12.1} {:>12.1} {:>9.2}x",
                r.image_size, r.sigma, r.input_pattern, r.algorithm, r.naive_us, r.opt_us,
                r.speedup
            );
        }
    }

    // Print best and worst speedups
    println!("\n--- Top 10 Best Speedups ---");
    let mut sorted_by_speedup: Vec<&BenchResult> = summary_results.iter().copied().collect();
    sorted_by_speedup.sort_by(|a, b| b.speedup.partial_cmp(&a.speedup).unwrap());
    for r in sorted_by_speedup.iter().take(10) {
        println!(
            "  {:<12} sigma={:<8} {:<10} {:<10} {:.2}x",
            r.image_size, r.sigma, r.input_pattern, r.algorithm, r.speedup
        );
    }

    println!("\n--- Top 10 Worst Speedups ---");
    sorted_by_speedup.reverse();
    for r in sorted_by_speedup.iter().take(10) {
        println!(
            "  {:<12} sigma={:<8} {:<10} {:<10} {:.2}x",
            r.image_size, r.sigma, r.input_pattern, r.algorithm, r.speedup
        );
    }
}
