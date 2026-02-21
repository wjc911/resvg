// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{ImageRefMut, f32_bound};
use usvg::filter::{ComponentTransfer, TransferFunction};

/// A 256-entry lookup table mapping each possible u8 input to its u8 output.
type Lut = [u8; 256];

/// Pre-computes a 256-entry lookup table for the given transfer function.
///
/// For each input byte value 0..=255, we compute the exact same floating-point
/// arithmetic that `transfer_scalar` would compute, then store the u8 result.
/// This is bit-exact with the per-pixel path because the inputs are always u8
/// values (0..=255) and the computation is deterministic.
fn build_lut(func: &TransferFunction) -> Lut {
    let mut lut: Lut = [0u8; 256];
    for i in 0u16..=255 {
        lut[i as usize] = transfer_scalar(func, i as u8);
    }
    lut
}

/// Applies component transfer functions for each `src` image channel.
///
/// Input image pixels should have an **unpremultiplied alpha**.
///
/// This implementation pre-computes a 256-entry lookup table for each
/// non-identity channel. This eliminates expensive per-pixel operations
/// (especially `powf` in the Gamma case) by replacing them with a single
/// table lookup per pixel per channel.
///
/// For very small images (fewer than 256 pixels), the LUT setup cost
/// (up to 1024 `transfer_scalar()` calls) exceeds the per-pixel savings,
/// so we fall back to direct per-pixel `transfer_scalar()` calls.
pub fn apply(fe: &ComponentTransfer, src: ImageRefMut) {
    let func_r = fe.func_r();
    let func_g = fe.func_g();
    let func_b = fe.func_b();
    let func_a = fe.func_a();

    let r_active = !is_dummy(func_r);
    let g_active = !is_dummy(func_g);
    let b_active = !is_dummy(func_b);
    let a_active = !is_dummy(func_a);

    // Fast path: nothing to do.
    if !r_active && !g_active && !b_active && !a_active {
        return;
    }

    // For small images, the LUT setup cost (256 transfer_scalar() calls per
    // active channel) exceeds the per-pixel savings. Fall back to direct
    // per-pixel computation.
    if src.data.len() < 256 {
        for pixel in src.data {
            if r_active {
                pixel.r = transfer_scalar(func_r, pixel.r);
            }
            if g_active {
                pixel.g = transfer_scalar(func_g, pixel.g);
            }
            if b_active {
                pixel.b = transfer_scalar(func_b, pixel.b);
            }
            if a_active {
                pixel.a = transfer_scalar(func_a, pixel.a);
            }
        }
        return;
    }

    // Pre-compute LUTs only for active channels.
    let lut_r = if r_active {
        build_lut(func_r)
    } else {
        IDENTITY_LUT
    };
    let lut_g = if g_active {
        build_lut(func_g)
    } else {
        IDENTITY_LUT
    };
    let lut_b = if b_active {
        build_lut(func_b)
    } else {
        IDENTITY_LUT
    };
    let lut_a = if a_active {
        build_lut(func_a)
    } else {
        IDENTITY_LUT
    };

    // Apply LUTs. Table lookups are gather operations that prevent
    // auto-vectorization, but they still outperform per-pixel floating-point
    // arithmetic (especially `powf`) for images with >= 256 pixels.
    // When only some channels are active, the identity LUT is a no-op in
    // terms of correctness (lut[x] == x for identity), so we can always
    // apply all four without branching in the hot loop.
    for pixel in src.data {
        pixel.r = lut_r[pixel.r as usize];
        pixel.g = lut_g[pixel.g as usize];
        pixel.b = lut_b[pixel.b as usize];
        pixel.a = lut_a[pixel.a as usize];
    }
}

/// An identity LUT where lut[i] == i for all i.
/// Computed at compile time for zero runtime cost.
const IDENTITY_LUT: [u8; 256] = {
    let mut lut = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        lut[i as usize] = i as u8;
        i += 1;
    }
    lut
};

/// Original naive implementation preserved verbatim for correctness testing.
#[cfg(test)]
fn apply_naive(fe: &ComponentTransfer, src: ImageRefMut) {
    for pixel in src.data {
        if !is_dummy(fe.func_r()) {
            pixel.r = transfer_scalar(fe.func_r(), pixel.r);
        }

        if !is_dummy(fe.func_b()) {
            pixel.b = transfer_scalar(fe.func_b(), pixel.b);
        }

        if !is_dummy(fe.func_g()) {
            pixel.g = transfer_scalar(fe.func_g(), pixel.g);
        }

        if !is_dummy(fe.func_a()) {
            pixel.a = transfer_scalar(fe.func_a(), pixel.a);
        }
    }
}

fn is_dummy(func: &TransferFunction) -> bool {
    match func {
        TransferFunction::Identity => true,
        TransferFunction::Table(values) => values.is_empty(),
        TransferFunction::Discrete(values) => values.is_empty(),
        TransferFunction::Linear { .. } => false,
        TransferFunction::Gamma { .. } => false,
    }
}

/// Computes the transfer function result for a single u8 input value.
///
/// This is the scalar per-pixel computation. It is used both by `apply_naive`
/// (the original path) and by `build_lut` to pre-compute the lookup table.
/// Since `build_lut` calls this with the exact same u8 inputs that would be
/// encountered at runtime, the LUT path is bit-exact with the scalar path.
fn transfer_scalar(func: &TransferFunction, c: u8) -> u8 {
    let c = c as f32 / 255.0;
    let c = match func {
        TransferFunction::Identity => c,
        TransferFunction::Table(values) => {
            let n = values.len() - 1;
            let k = (c * (n as f32)).floor() as usize;
            let k = std::cmp::min(k, n);
            if k == n {
                values[k]
            } else {
                let vk = values[k];
                let vk1 = values[k + 1];
                let k = k as f32;
                let n = n as f32;
                vk + (c - k / n) * n * (vk1 - vk)
            }
        }
        TransferFunction::Discrete(values) => {
            let n = values.len();
            let k = (c * (n as f32)).floor() as usize;
            values[std::cmp::min(k, n - 1)]
        }
        TransferFunction::Linear { slope, intercept } => slope * c + intercept,
        TransferFunction::Gamma {
            amplitude,
            exponent,
            offset,
        } => amplitude * c.powf(*exponent) + offset,
    };

    (f32_bound(0.0, c, 1.0) * 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use rgb::RGBA8;

    /// Helper to create an ImageRefMut from a mutable slice.
    fn make_image(data: &mut [RGBA8]) -> ImageRefMut<'_> {
        let len = data.len() as u32;
        ImageRefMut::new(len, 1, data)
    }

    /// Test that the LUT-based `apply` and the naive `apply_naive` produce
    /// bit-exact results for every possible u8 input on all transfer function types.
    fn assert_bit_exact(func: &TransferFunction) {
        // Build test data: all 256 possible values for every channel
        let mut data_lut: Vec<RGBA8> = (0u16..=255)
            .map(|i| {
                let v = i as u8;
                RGBA8 {
                    r: v,
                    g: v,
                    b: v,
                    a: v,
                }
            })
            .collect();
        let mut data_naive = data_lut.clone();

        let fe = make_component_transfer(func.clone());

        apply(&fe, make_image(&mut data_lut));
        apply_naive(&fe, make_image(&mut data_naive));

        for i in 0..256 {
            assert_eq!(
                data_lut[i], data_naive[i],
                "Mismatch at input {}: lut={:?} naive={:?}",
                i, data_lut[i], data_naive[i]
            );
        }
    }

    /// Helper to create a ComponentTransfer with the same function on all channels.
    fn make_component_transfer(func: TransferFunction) -> ComponentTransfer {
        ComponentTransfer::new(
            usvg::filter::Input::SourceGraphic,
            func.clone(),
            func.clone(),
            func.clone(),
            func,
        )
    }

    #[test]
    fn bit_exact_identity() {
        assert_bit_exact(&TransferFunction::Identity);
    }

    #[test]
    fn bit_exact_table() {
        assert_bit_exact(&TransferFunction::Table(vec![0.0, 0.5, 1.0]));
    }

    #[test]
    fn bit_exact_table_many() {
        assert_bit_exact(&TransferFunction::Table(vec![
            0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0,
        ]));
    }

    #[test]
    fn bit_exact_discrete() {
        assert_bit_exact(&TransferFunction::Discrete(vec![0.0, 0.33, 0.67, 1.0]));
    }

    #[test]
    fn bit_exact_linear() {
        assert_bit_exact(&TransferFunction::Linear {
            slope: 0.5,
            intercept: 0.25,
        });
    }

    #[test]
    fn bit_exact_gamma_standard() {
        assert_bit_exact(&TransferFunction::Gamma {
            amplitude: 1.0,
            exponent: 2.2,
            offset: 0.0,
        });
    }

    #[test]
    fn bit_exact_gamma_with_offset() {
        assert_bit_exact(&TransferFunction::Gamma {
            amplitude: 0.8,
            exponent: 0.45,
            offset: 0.1,
        });
    }

    #[test]
    fn bit_exact_gamma_inverse() {
        assert_bit_exact(&TransferFunction::Gamma {
            amplitude: 1.0,
            exponent: 1.0 / 2.2,
            offset: 0.0,
        });
    }
}
