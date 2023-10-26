
#[inline]
pub unsafe fn div(
    x1: *const f32,
    x2: *const f32,
    y: *mut f32,
    len: usize,
) {
    for i in 0..len {
        *y.add(i) = *x1.add(i) / *x2.add(i)
    }
}

#[inline]
pub unsafe fn div_wrt_a(
    x2: *const f32,
    gy: *const f32,
    g1: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let g1ptr = g1.add(i);
        *g1ptr = (*g1ptr * beta) + (*gy.add(i) * (1. / *x2.add(i)));
    }
}

#[inline]
pub unsafe fn div_wrt_b(
    x1: *const f32,
    x2: *const f32,
    gy: *const f32,
    g2: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let g2ptr = g2.add(i);
        *g2ptr = (*g2ptr * beta) + (*gy.add(i) * (*x1.add(i) / (*x2.add(i)).powi(2)))
    }
}

