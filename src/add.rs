
#[inline]
pub unsafe fn add(
    x1: *const f32,
    x2: *const f32,
    y: *mut f32,
    len: usize,
) {
    for i in 0..len {
        *y.add(i) = *x1.add(i) + *x2.add(i);
    }
}

#[inline]
pub unsafe fn add_wrt_x1(
    gy: *const f32,
    g1: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let gaptr = g1.add(i);
        *gaptr = (*gaptr * beta) + *gy.add(i)
    }
}

#[inline]
pub unsafe fn add_wrt_x2(
    gy: *const f32,
    g2: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let gbptr = g2.add(i);
        *gbptr = (*gbptr * beta) + *gy.add(i)
    }
}