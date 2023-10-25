
/// ## Inputs
/// - A: Input (NCHW)
/// - B: Values to Add to A (AXIS x 1)
/// - C: Output (NCHW)
/// - Dim: Dimensions of C
/// - Axis: Axis to iterate
#[inline]
pub unsafe fn axis_sub(
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

                    *cptr.add(i) = *a.add(i) - *b.add(indices[axis]);
                }
            }
        }
    }
}

#[inline]
pub unsafe fn axis_sub_wrt_a(
    gc: *const f32,
    ga: *mut f32,
    dim: [usize; 4],
    axis: usize,
    beta: f32,
) {
    let len = dim[0] * dim[1] * dim[2] * dim[3];

    for i in 0..len {
        let gaptr = ga.add(i);
        *gaptr = (*gaptr * beta) + *gc.add(i)
    }
}

#[inline]
pub unsafe fn axis_sub_wrt_b(
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

                    *gb.add(indices[axis]) -= *gc.add(i)
                }
            }
        }
    }
}

