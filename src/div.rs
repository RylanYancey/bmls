
use itertools::izip;
use crate::error::BMLSError;
use crate::error;

#[inline]
pub fn div(
    x1: &[f32],
    x2: &[f32],
    y: &mut [f32],
) -> Result<(), BMLSError> {
    if x1.len() != y.len() {
        return error::length_mismatch("X1", x1.len(), "Y", y.len())
    }

    if x2.len() != y.len() {
        return error::length_mismatch("X2", x2.len(), "Y", y.len())
    }

    for (x1, x2, y) in izip!(x1, x2, y) {
        *y = *x1 / *x2
    }

    Ok(())
}

#[inline]
pub fn div_wrt_x1(
    x2: &[f32],
    gy: &[f32],
    g1: &mut [f32],
) -> Result<(), BMLSError> {
    if x2.len() != gy.len() {
        return error::length_mismatch("X2", x2.len(), "GY", gy.len())
    }

    if g1.len() != gy.len() {
        return error::length_mismatch("G1", g1.len(), "GY", gy.len())
    }

    for (x2, gy, g1) in izip!(x2, gy, g1) {
        *g1 += *gy * (1. / *x2);
    }

    Ok(())
}

#[inline]
pub fn div_wrt_x2(
    x1: &[f32],
    x2: &[f32],
    gy: &[f32],
    g2: &mut [f32],
) -> Result<(), BMLSError> {
    if x1.len() != gy.len() {
        return error::length_mismatch("X1", x1.len(), "GY", gy.len())
    }

    if x2.len() != gy.len() {
        return error::length_mismatch("X2", x2.len(), "GY", gy.len())
    }

    if g2.len() != gy.len() {
        return error::length_mismatch("G2", g2.len(), "GY", gy.len())
    }


    for (x1, x2, gy, g2) in izip!(x1, x2, gy, g2) {
        *g2 += *gy * (*x1 / (*x2 * *x2));
    }

    Ok(())
}

