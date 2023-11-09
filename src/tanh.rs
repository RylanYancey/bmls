
use itertools::izip;
use crate::error::BMLSError;
use crate::error;

#[inline]
pub fn tanh(
    x: &[f32],
    y: &mut [f32]
) -> Result<(), BMLSError> {
    if x.len() != y.len() {
        return error::length_mismatch("X", x.len(), "Y", y.len())
    }

    for (x, y) in izip!(x, y) {
        let posex = f32::exp(*x);
        let negex = f32::exp(-*x);
        // y = (e^z - e^-z) / (e^z + e^-z)
        *y = (posex - negex) / (posex + negex)
    }

    Ok(())
}

#[inline]
pub fn tanh_wrt_x(
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
        // gx = gy * (1 - y^2)
        *gx += *gy * (1. - *y * *y)
    }

    Ok(())
}