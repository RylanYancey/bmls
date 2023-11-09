
use itertools::izip;
use crate::error::BMLSError;
use crate::error;

/// # Dropout Operator
/// - X: Input
/// - R: Input to be filled with random values
/// - Y: Output 
/// - Rate: Dropout Rate
#[inline]
pub fn dropout(
    x: &[f32],
    r: &mut [f32],
    y: &mut [f32],
    rate: f32,
) -> Result<(), BMLSError> {
    if x.len() != y.len() {
        return error::length_mismatch("X", x.len(), "Y", y.len())
    }

    if r.len() != y.len() {
        return error::length_mismatch("R", r.len(), "Y", y.len())
    }

    if rate > 1.0 || rate < 0.0 {
        return error::invalid_dropout_rate(rate)
    }

    let factor = 1. / (1. - rate);
    for (x, r, y) in izip!(x, r, y) {
        *r = fastrand::f32();

        if *r < rate {
            *y = 0.0;
        } else {
            *y = *x * factor
        }
    }

    Ok(())
}

/// # Dropout w.r.t. X
/// - R: Random values generated in the forward op.
/// - GY: Gradient w.r.t. Y
/// - GX: Gradient w.r.t. X
/// - rate: Dropout Rate
#[inline]
pub fn dropout_wrt_x(
    r: &[f32],
    gy: &[f32],
    gx: &mut [f32],
    rate: f32,
) -> Result<(), BMLSError> {
    if gx.len() != gy.len() {
        return error::length_mismatch("GX", gx.len(), "GY", gy.len())
    }

    if r.len() != gy.len() {
        return error::length_mismatch("R", r.len(), "GY", gy.len())
    }

    if rate > 1.0 || rate < 0.0 {
        return error::invalid_dropout_rate(rate)
    }

    let factor = 1. / (1. - rate);
    for (r, gy, gx) in izip!(r, gy, gx) {
        if *r < rate {
            *gx += *gy * factor
        }
    }

    Ok(())
}