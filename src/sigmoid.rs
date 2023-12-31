
use itertools::izip;
use crate::error::BMLSError;
use crate::error;

/// # Sigmoid Operation
/// - X: Input
/// - Y: Output
#[inline]
pub fn sigmoid(
    x: &[f32],
    y: &mut [f32]
) -> Result<(), BMLSError> {
    if x.len() != y.len() {
        return error::length_mismatch("X", x.len(), "Y", y.len())
    }

    for (x, y) in izip!(x, y) {
        *y = 1. / (1. + f32::exp(-*x))
    }

    Ok(())
}

/// # Sigmoid w.r.t. X
/// - Y: Output of the forward op
/// - Gy: Gradient w.r.t. Y. 
/// - Gx: Gradient w.r.t. X. 
#[inline]
pub fn sigmoid_wrt_x(
    y: &[f32],
    gy: &[f32],
    gx: &mut [f32]
) -> Result<(), BMLSError> {
    if y.len() != gx.len() {
        return error::length_mismatch("Y", y.len(), "GX", gx.len())
    }

    if gy.len() != gx.len() {
        return error::length_mismatch("GY", gy.len(), "GX", gx.len())
    }

    for (y, gy, gx) in izip!(y, gy, gx) {
        *gx += *gy * (*y * (1. - *y));
    }

    Ok(())
}