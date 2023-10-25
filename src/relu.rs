
#[inline]
pub unsafe fn relu(
    x: *const f32,
    y: *mut f32,
    len: usize,
) {
    for i in 0..len {
        let xptr = x.add(i);
        if *x > 0.0 {
            *y.add(i) = *x;
        } else {
            *y.add(i) = 0.;
        }
    }
}

#[inline]
pub unsafe fn relu_wrt_x(
    x: *const f32,
    gy: *const f32,
    g1: *mut f32,
    len: usize,
    beta: f32,
) {
    for i in 0..len {
        let g1ptr = g1.add(i);
        *g1ptr *= beta;

        if *x.add(i) > 0. {
            *g1ptr += *gy.add(i)
        } 
    }
}