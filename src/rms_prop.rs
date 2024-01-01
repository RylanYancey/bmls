
use itertools::izip;
use crate::error::BMLSError;
use crate::error;

/// # RMS_Prop Optimizer
/// - G: Gradient w.r.t. W.
/// - S: Exponentially weighted average of Past squares of gradients
/// - W: Weight Tensor
/// - LR: Learning Rate
/// - Beta: Hyperparameter
#[inline]
pub fn rms_prop(
    g: &[f32],
    s: &mut [f32],
    w: &mut [f32],
    lr: f32,
    beta: f32,
) -> Result<(), BMLSError> {
    if g.len() != s.len() {
        return error::length_mismatch("G", g.len(), "V", s.len())
    }

    if s.len() != w.len() {
        return error::length_mismatch("S", s.len(), "W", w.len())
    }

    for (g, s, w) in izip!(g, s, w) {
        // v = Bv + (1 - B)g
        *s = *s * beta + (1. - beta) * f32::powi(*g, 2);
        // w -= lr * v
        *w -= lr * (*w / (f32::sqrt(*s) + 0.00000000001));
    }

    Ok(())
}