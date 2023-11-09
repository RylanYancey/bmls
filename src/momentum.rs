
use itertools::izip;
use crate::error::BMLSError;
use crate::error;

/// # Momentum Optimizer
/// - G: Gradient w.r.t. W.
/// - P: Exponentially weighted average of Past gradients
/// - W: Weight Tensor
/// - LR: Learning Rate
/// - Beta: Hyperparameter
#[inline]
pub fn momentum(
    g: &[f32],
    v: &mut [f32],
    w: &mut [f32],
    lr: f32,
    beta: f32,
) -> Result<(), BMLSError> {
    if g.len() != v.len() {
        return error::length_mismatch("G", g.len(), "V", v.len())
    }

    if v.len() != w.len() {
        return error::length_mismatch("V", v.len(), "W", w.len())
    }

    for (g, v, w) in izip!(g, v, w) {
        // v = Bv + (1 - B)g
        *v = *v * beta + (1. - beta) * *g;
        // w -= lr * v
        *w -= lr * *v;
    }

    Ok(())
}