
/// # Multiplication Operation
/// - X1: Left Operand
/// - X2: Right Operand
/// - Y: Output
/// - Len: Length of X1, X2, and Y. 
#[inline]
pub unsafe fn mul(
    x1: *const f32,
    x2: *const f32,
    y: *mut f32,
    len: usize,
) {
    for i in 0..len {
        *y.add(i) = *x1.add(i) * *x2.add(i);
    }
}

/// # Multiplication W.r.t. A
/// - X2: Input 2 in the forward op.
/// - GY: Gradient w.r.t. Output Y
/// - G1: Gradient W.r.t. Input X1
/// - Len: Length of GY, G1. 
/// - Beta: Scaling factor for g1. 
#[inline]
pub unsafe fn mul_wrt_a(
    x2: *const f32,
    gy: *const f32,
    g1: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let g1ptr = g1.add(i);
        *g1ptr = (*g1ptr * beta) + (*gy.add(i) * *x2.add(i))
    }
}

/// # Multiplication w.r.t. B
/// - X1: Input 1 in the forward op.
/// - GY: Gradient w.r.t. Output Y
/// - G2: Gradient w.r.t. Input X2
/// - Len: Length of GY, G2, 
/// - Beta: Scaling factor for G2. 
#[inline]
pub unsafe fn mul_wrt_b(
    x1: *const f32,
    gy: *const f32,
    g2: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let g2ptr = g2.add(i);
        *g2ptr = (*g2ptr * beta) + (*gy.add(i) * *x1.add(i))
    }
}