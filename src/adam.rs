
use itertools::izip;
use crate::error::BMLSError;
use crate::error;

/// # Adam Optimizer
/// - G: Gradient w.r.t. W
/// - V: Exponentialy weighted average of past gradients
/// - S: Exponentialy weighted average of past squares of gradients
/// - W: Weight Tensor
/// - Lr: Learning Rate
/// - Beta1: Hyperparameter,
/// - Beta2: Hyperparameter,
#[inline]
pub fn adam(
    g: &[f32],
    v: &mut [f32],
    s: &mut [f32],
    w: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
) -> Result<(), BMLSError> {
    if g.len() != v.len() {
        return error::length_mismatch("G", g.len(), "V", v.len())
    }

    if v.len() != s.len() {
        return error::length_mismatch("V", v.len(), "S", s.len())
    }

    if s.len() != w.len() {
        return error::length_mismatch("S", s.len(), "W", w.len())
    }

    for (g, v, s, w) in izip!(g, v, s, w) {
        // update V and S
        *v = (*v * beta1) + (1. - beta1) * g;
        *s = (*s * beta2) + (1. - beta2) * (g * g);

        // correct V and S
        *v /= 1. - beta1;
        *s /= 1. - beta2;

        // update the weights
        // w -= lr * (v / (sqrt(s) + e))
        *w -= lr * (*v / (f32::sqrt(*s) + 0.000000000000001));
    }

    Ok(())   
}