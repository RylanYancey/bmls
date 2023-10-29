
/// # Adam Optimizer
/// - G: Gradient w.r.t. W
/// - V: Exponentialy weighted average of past gradients
/// - S: Exponentialy weighted average of past squares of gradients
/// - W: Weight Tensor
/// - Lr: Learning Rate
/// - Beta1: Hyperparameter,
/// - Beta2: Hyperparameter,
/// - Len: Length of G, V, S, and W. 
#[inline]
pub unsafe fn adam(
    g: *const f32,
    v: *mut f32,
    s: *mut f32,
    w: *mut f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    len: usize,
) {
    for i in 0..len {
        let g = g.add(i);
        // assign V
        let v = v.add(i);
        *v = (*v * beta1) + (1. - beta1) * *g;

        // assign S
        let s = s.add(i);
        *s = (*s * beta2) + (1. - beta2) * (*g * *g);

        // correct V and S
        *v /= 1. - beta1;
        *s /= 1. - beta2;

        // update the weights
        *w.add(i) -= lr * (*v / (f32::sqrt(*s) + 0.00000000001))
    }
}