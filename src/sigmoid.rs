
#[inline]
pub unsafe fn sigmoid(
    a: *const f32,
    b: *mut f32,
    len: usize,
) {
    for i in 0..len {
        *b.add(i) = 1.0 / (1.0 + (-*a.add(i)).exp())
    }
}

#[inline]
pub unsafe fn sigmoid_wrt_a(
    b: *const f32,
    gb: *const f32,
    ga: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let gaptr = ga.add(i);
        *gaptr = (*gaptr * beta) + (*gb.add(i) * (*b.add(i) * (1. - *b.add(i))));
    }
}