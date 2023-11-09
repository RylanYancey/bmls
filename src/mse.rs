
use crate::error::BMLSError;
use crate::error;

/// ## Inputs
/// - T: Truth Label (N x C)
/// - P: Prediction (N x C)
/// - E: Error (N x 1) vector
/// - G: Gradient (N x C)
#[inline]
pub fn mse(
    t: &[f32],
    p: &[f32],
    e: &mut [f32],
    g: &mut [f32],
    dim: [usize; 2],
) -> Result<(), BMLSError> {
    let len = dim[0] * dim[1];
    if t.len() != len {
        return error::length_mismatch("T", t.len(), "Dim", len)
    }

    if t.len() != p.len() {
        return error::length_mismatch("T", t.len(), "P", p.len())
    }

    if e.len() != dim[0] {
        return error::length_mismatch("E", e.len(), "Dim[0]", dim[0])
    }

    if g.len() != p.len() {
        return error::length_mismatch("G", g.len(), "P", p.len())
    }

    let rows = dim[0];
    let cols = dim[1];
    
    for i in 0..rows {
        let mut sum = 0.0;

        for j in 0..cols {
            let index = i * cols + j;
            let predicted = p[index];
            let target = t[index];

            sum += f32::powi(predicted - target, 2);
            g[index] = 2.0 * (predicted - target);
        }

        e[i] = sum / cols as f32
    }

    Ok(())
}
