
/// # Sigmoid Operation
/// - X: Input
/// - Y: Output
/// - Len: Length of X and Y
#[inline]
pub unsafe fn sigmoid(
    x: *const f32,
    y: *mut f32,
    len: usize,
) {
    for i in 0..len {
        *y.add(i) = 1.0 / (1.0 + (-*x.add(i)).exp())
    }
}

/// # Sigmoid w.r.t. X
/// - Y: Output in the forward op
/// - Gy: Gradient w.r.t. Y. 
/// - Gx: Gradient w.r.t. X. 
/// - Len: Length of X,GY, and GX. 
/// - Beta: Scaling factor for GX. 
#[inline]
pub unsafe fn sigmoid_wrt_x(
    y: *const f32,
    gy: *const f32,
    gx: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let gxptr = gx.add(i);
        *gxptr = (*gxptr * beta) + (*gy.add(i) * (*y.add(i) * (1. - *y.add(i))));
    }
}