
#[inline]
pub unsafe fn div(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    len: usize,
) {
    for i in 0..len {
        *c.add(i) = *a.add(i) / *b.add(i)
    }
}

#[inline]
pub unsafe fn div_wrt_a(
    b: *const f32,
    gc: *const f32,
    ga: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let gaptr = ga.add(i);
        *gaptr = (*gaptr * beta) + (*gc.add(i) * (1. / *b.add(i)));
    }
}

#[inline]
pub unsafe fn div_wrt_b(
    a: *const f32,
    b: *const f32,
    gc: *const f32,
    gb: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let gbptr = gb.add(i);
        *gbptr = (*gbptr * beta) + (*gc.add(i) * (*a.add(i) / (*b.add(i)).powi(2)))
    }
}

