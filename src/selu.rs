

use itertools::izip;
use crate::error::BMLSError;
use crate::error;

/// Scaled Exponential Linear Unit
/// - X: Input
/// - Y: Output
/// - A: Alpha (pos/neg scaling factor ~1.6733)
/// - L: Lambda (negative scaling factor ~1.0507)
#[inline]
pub fn selu(
    x: &[f32],
    y: &mut [f32],
    a: f32,
    l: f32,
) -> Result<(), BMLSError> {
    if x.len() != y.len() {
        return error::length_mismatch("X", x.len(), "Y", y.len())
    }

    for (x, y) in izip!(x, y) {
        if *x > 0.0 {
            *y = l* *x;
        } else {
            *y = l * a * (f32::exp(*x) - 1.);
        }
    }

    Ok(())
}

#[inline]
pub fn selu_wrt_x(
    x: &[f32],
    gy: &[f32],
    g1: &mut [f32],
    a: f32,
    l: f32,
) -> Result<(), BMLSError> {
    if x.len() != g1.len() {
        return error::length_mismatch("X", x.len(), "G1", g1.len())
    }    

    if gy.len() != g1.len() {
        return error::length_mismatch("GY", gy.len(), "G1", g1.len())
    }

    for (x, gy, g1) in izip!(x, gy, g1) {
        if *x > 0.0 {
            *g1 += l * *gy;
        } else {
            *g1 += l*a*f32::exp(*x)
        }
    }

    Ok(())
}