
use itertools::izip;
use crate::error::BMLSError;
use crate::error;

#[inline]
pub fn add(
    x1: &[f32],
    x2: &[f32],
    y: &mut [f32],
) -> Result<(), BMLSError> {
    if x1.len() != x2.len() {
        return error::length_mismatch("X2", x1.len(), "X2", x2.len())
    }

    if x2.len() != y.len() {
        return error::length_mismatch("X2", x2.len(), "Y", y.len())
    }

    for (x1, x2, y) in izip!(x1, x2, y) {
        *y = *x1 + *x2;
    }

    Ok(())
}

#[inline]
pub fn add_wrt_x1(
    gy: &[f32],
    g1: &mut [f32],
) -> Result<(), BMLSError> {
    if gy.len() != g1.len() {
        return error::length_mismatch("GY", gy.len(), "G1", g1.len())
    }

    for (gy, g1) in izip!(gy, g1) {
        *g1 += *gy;
    }

    Ok(())
}

#[inline]
pub fn add_wrt_x2(
    gy: &[f32],
    g2: &mut [f32],
) -> Result<(), BMLSError> {
    if gy.len() != g2.len() {
        return error::length_mismatch("GY", gy.len(), "G2", g2.len())
    }

    for (gy, g2) in izip!(gy, g2) {
        *g2 += *gy;
    }

    Ok(())
}