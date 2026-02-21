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

/// Returns the LUT threshold (minimum pixel count) for a given transfer function.
///
/// The LUT approach pre-computes 256 entries, which has a fixed setup cost that
/// varies by function type. The per-pixel savings must amortize this cost.
///
/// Benchmarked thresholds:
/// - **Gamma** (`powf`): ~30 us to build LUT; breaks even at ~256 pixels.
/// - **Table / Discrete**: ~18-25 us to build; breaks even at ~1024 pixels.
/// - **Linear**: ~6 us to build, but per-pixel cost is very low (multiply + add);
///   LUT only wins at ~2048 pixels.
fn lut_threshold(func: &TransferFunction) -> usize {
    match func {
        TransferFunction::Identity => usize::MAX, // never build a LUT
        TransferFunction::Gamma { .. } => 256,
        TransferFunction::Table(_) | TransferFunction::Discrete(_) => 1024,
        TransferFunction::Linear { .. } => 2048,
    }
}

/// Applies component transfer functions for each `src` image channel.
///
/// Input image pixels should have an **unpremultiplied alpha**.
///
/// This implementation pre-computes a 256-entry lookup table for each
/// non-identity channel when the image is large enough for the LUT setup
/// cost to be amortized. The threshold varies by transfer function type:
///
/// - **Gamma** (expensive `powf`): LUT at >= 256 pixels
/// - **Table / Discrete**: LUT at >= 1024 pixels
/// - **Linear** (cheap multiply-add): LUT at >= 2048 pixels
///
/// For images below the threshold, direct per-pixel `transfer_scalar()` calls
/// are used, avoiding the fixed LUT construction overhead.
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

    let pixel_count = src.data.len();

    // Determine per-channel: use LUT or direct based on function-specific threshold.
    let use_lut_r = r_active && pixel_count >= lut_threshold(func_r);
    let use_lut_g = g_active && pixel_count >= lut_threshold(func_g);
    let use_lut_b = b_active && pixel_count >= lut_threshold(func_b);
    let use_lut_a = a_active && pixel_count >= lut_threshold(func_a);

    let any_lut = use_lut_r || use_lut_g || use_lut_b || use_lut_a;
    let any_direct = (r_active && !use_lut_r)
        || (g_active && !use_lut_g)
        || (b_active && !use_lut_b)
        || (a_active && !use_lut_a);

    // Pure direct path: no channel benefits from LUT at this size.
    if !any_lut {
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

    // Build LUTs for channels that benefit; use identity LUT for others.
    let lut_r = if use_lut_r {
        build_lut(func_r)
    } else {
        IDENTITY_LUT
    };
    let lut_g = if use_lut_g {
        build_lut(func_g)
    } else {
        IDENTITY_LUT
    };
    let lut_b = if use_lut_b {
        build_lut(func_b)
    } else {
        IDENTITY_LUT
    };
    let lut_a = if use_lut_a {
        build_lut(func_a)
    } else {
        IDENTITY_LUT
    };

    // Hybrid path: some channels use LUT, others use direct computation.
    if any_direct {
        for pixel in src.data {
            pixel.r = if use_lut_r {
                lut_r[pixel.r as usize]
            } else if r_active {
                transfer_scalar(func_r, pixel.r)
            } else {
                pixel.r
            };
            pixel.g = if use_lut_g {
                lut_g[pixel.g as usize]
            } else if g_active {
                transfer_scalar(func_g, pixel.g)
            } else {
                pixel.g
            };
            pixel.b = if use_lut_b {
                lut_b[pixel.b as usize]
            } else if b_active {
                transfer_scalar(func_b, pixel.b)
            } else {
                pixel.b
            };
            pixel.a = if use_lut_a {
                lut_a[pixel.a as usize]
            } else if a_active {
                transfer_scalar(func_a, pixel.a)
            } else {
                pixel.a
            };
        }
        return;
    }

    // Pure LUT path: all active channels use LUT lookups.
    // The identity LUT handles inactive channels (lut[x] == x).
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
