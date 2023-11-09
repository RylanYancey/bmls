
use itertools::izip;
use crate::error::BMLSError;
use crate::error;

/// # Multiplication Operation
/// - X1: Left Operand
/// - X2: Right Operand
/// - Y: Output
/// - Len: Length of X1, X2, and Y. 
#[inline]
pub fn mul(
    x1: &[f32],
    x2: &[f32],
    y: &mut [f32],
) -> Result<(), BMLSError> {
    if x1.len() != x2.len() {
        return error::length_mismatch("X1", x1.len(), "X2", x2.len());
    }

    if x2.len() != y.len() {
        return error::length_mismatch("X2", x2.len(), "Y", y.len());
    }

    for (x1, x2, y) in izip!(x1, x2, y) {
        *y = *x1 * *x2;
    }

    Ok(())
}

/// # Multiplication W.r.t. X1
/// - X2: Input 2 in the forward op.
/// - GY: Gradient w.r.t. Output Y
/// - G1: Gradient W.r.t. Input X1
#[inline]
pub fn mul_wrt_x1(
    x2: &[f32],
    gy: &[f32],
    g1: &mut [f32],
) -> Result<(), BMLSError> {
    if x2.len() != gy.len() {
        return error::length_mismatch("X2", x2.len(), "GY", gy.len());
    }

    if x2.len() != g1.len() {
        return error::length_mismatch("X2", x2.len(), "G1", g1.len());
    }

    for (x2, gy, g1) in izip!(x2, gy, g1) {
        *g1 += *x2 * *gy;
    }

    Ok(())
}

/// # Multiplication w.r.t. X2
/// - X1: Input 1 in the forward op.
/// - GY: Gradient w.r.t. Output Y
/// - G2: Gradient w.r.t. Input X2
#[inline]
pub fn mul_wrt_x2(
    x1: &[f32],
    gy: &[f32],
    g2: &mut [f32]
) -> Result<(), BMLSError> {
    if x1.len() != gy.len() {
        return error::length_mismatch("X1", x1.len(), "GY", gy.len());
    }

    if x1.len() != g2.len() {
        return error::length_mismatch("X1", x1.len(), "G2", g2.len());
    }

    for (x1, gy, g2) in izip!(x1, gy, g2) {
        *g2 += *x1 * *gy;
    }

    Ok(())
}