
#[inline]
pub unsafe fn axis_div(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    dim: [usize; 4],
    axis: usize,
) {
    let cptr = c;

    for n in 0..dim[0] {
        for c in 0..dim[1] {
            for h in 0..dim[2] {
                for w in 0..dim[3] {
                    let i = n * dim[1] * dim[2] * dim[3] + c * dim[2] * dim[3] + h * dim[3] + w;
                    let indices = [n, c, h, w];

                    *cptr.add(i) = *a.add(i) / *b.add(indices[axis]);
                }
            }
        }
    }
}

#[inline]
pub unsafe fn axis_div_wrt_a(
    b: *const f32,
    gc: *const f32,
    ga: *mut f32,
    dim: [usize; 4],
    axis: usize,
    beta: f32,
) {
    for n in 0..dim[0] {
        for c in 0..dim[1] {
            for h in 0..dim[2] {
                for w in 0..dim[3] {
                    let i = n * dim[1] * dim[2] * dim[3] + c * dim[2] * dim[3] + h * dim[3] + w;
                    let indices = [n, c, h, w];

                    let gaptr = ga.add(i);
                    *gaptr = (*gaptr * beta) + (*gc.add(i) * (1. / *b.add(indices[axis])))
                }
            }
        }
    }
}

#[inline]
pub unsafe fn axis_div_wrt_b(
    a: *const f32,
    b: *const f32,
    gc: *const f32,
    gb: *mut f32,
    dim: [usize; 4],
    axis: usize,
    beta: f32,
) {
    for i in 0..dim[axis] {
        *gb.add(i) *= beta;
    }

    for n in 0..dim[0] {
        for c in 0..dim[1] {
            for h in 0..dim[2] {
                for w in 0..dim[3] {
                    let i = n * dim[1] * dim[2] * dim[3] + c * dim[2] * dim[3] + h * dim[3] + w;
                    let indices = [n, c, h, w];

                    *gb.add(i) += *gc.add(i) * (*a.add(i) / f32::powi(*b.add(indices[axis]), 2))
                }
            }
        }
    }
}