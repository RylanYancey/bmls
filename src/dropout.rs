
/// # Dropout Operator
/// - X: Input
/// - Y: Output
/// - Len: Length of X & Y. 
/// - Rate: Dropout Rate
#[inline]
pub unsafe fn dropout(
    x: *const f32,
    y: *mut f32,
    len: usize,
    rate: f32,
) {
    let factor = 1. / (1. - rate);

    for i in 0..len {
        if fastrand::f32() < rate {
            *y.add(i) = 0.0;
        } else {
            *y.add(i) = *x.add(i) * factor;
        }
    }
}

/// # Dropout w.r.t. X
/// - Y: Output in the forward op
/// - GY: Gradient w.r.t. Y
/// - GX: Gradient w.r.t. X
/// - len: Length of Y, GY, and GX. 
/// - rate: Dropout Rate
/// - Beta: scaling factor
#[inline]
pub unsafe fn dropout_wrt_x(
    y: *const f32,
    gy: *const f32,
    gx: *mut f32,
    len: usize,
    rate: f32,
    beta: f32,
) {
    let factor = 1. / (1. - rate);

    for i in 0..len {
        let gxptr = gx.add(i);

        if *y.add(i) != 0.0 {
            *gxptr += (*gxptr * beta) * *gy.add(i) * factor;
        }
    }
}