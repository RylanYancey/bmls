
use itertools::izip;
use crate::error::BMLSError;
use crate::error;

#[inline]
pub fn sgd(
    g: &[f32],
    w: &mut [f32],
    lr: f32,
) -> Result<(), BMLSError> {
    if g.len() != w.len() {
        return error::length_mismatch("G", g.len(), "W", w.len())
    }

    for (g, w) in izip!(g, w) {
        *w -= *g * lr
    }

    Ok(())
}