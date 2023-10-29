
/// # Momentum Optimizer
/// - G: Gradient w.r.t. W.
/// - P: Exponentially weighted average of Past gradients
/// - W: Weight Tensor
/// - LR: Learning Rate
/// - Beta: Hyperparameter
/// - Len: length of G, P, and W. 
#[inline]
pub unsafe fn momentum(
    g: *const f32,
    p: *mut f32,
    w: *mut f32,
    lr: f32,
    beta: f32,
    len: usize,
) {
    for i in 0..len {
        let g = g.add(i);
        *p = (*p * beta) + (1. - beta) * *g.add(i);
        *w.add(i) -= lr * *p;
    }
}