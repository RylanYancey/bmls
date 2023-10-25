
#[inline]
pub unsafe fn sub(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    len: usize,
) {
    for i in 0..len {
        *c.add(len) = *a.add(len) - *b.add(len);
    }
}

#[inline]
pub unsafe fn sub_wrt_a(
    gc: *const f32,
    ga: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let gaptr = ga.add(i);
        *gaptr = (*gaptr * beta) + *gc.add(i)
    }
}

#[inline]
pub unsafe fn sub_wrt_b(
    gc: *const f32,
    gb: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let gbptr = gb.add(i);
        *gbptr = (*gbptr * beta) - *gc.add(i)
    }
}