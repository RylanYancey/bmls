
#[inline]
pub unsafe fn relu(
    a: *const f32,
    b: *mut f32,
    len: usize,
) {
    for i in 0..len {
        if *a.add(i) > 0. {
            *b.add(i) = *a.add(i);
        } else {
            *b.add(i) = 0.;
        }
    }
}

#[inline]
pub unsafe fn relu_wrt_a(
    a: *const f32,
    gb: *const f32,
    ga: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let gaptr = ga.add(i);
        *gaptr *= beta;

        if *a.add(i) > 0. {
            *gaptr += *gb.add(i)
        } 
    }
}