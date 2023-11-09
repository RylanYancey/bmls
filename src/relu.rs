
use itertools::izip;
use crate::error::BMLSError;
use crate::error;

#[inline]
pub fn relu(
    x: &[f32],
    y: &mut [f32]
) -> Result<(), BMLSError> {
    if x.len() != y.len() {
        return error::length_mismatch("X", x.len(), "Y", y.len())
    }

    for (x, y) in izip!(x, y) {
        if *x > 0.0 {
            *y = *x;
        } else {
            *y = 0.;
        }
    }

    Ok(())
}

#[inline]
pub fn relu_wrt_x(
    x: &[f32],
    gy: &[f32],
    g1: &mut [f32],
) -> Result<(), BMLSError> {
    if x.len() != g1.len() {
        return error::length_mismatch("X", x.len(), "G1", g1.len())
    }    

    if gy.len() != g1.len() {
        return error::length_mismatch("GY", gy.len(), "G1", g1.len())
    }

    for (x, gy, g1) in izip!(x, gy, g1) {
        if *x > 0.0 {
            *g1 += *gy;
        }
    }

    Ok(())
}