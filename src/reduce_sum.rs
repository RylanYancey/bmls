
/// # Reduce Sum Operator
/// - X: Input
/// - Y: Output
/// - X_shape: shape of X
/// - Axis: Axis to sum
/// 
/// Y Shape is the same as X, but with the specified Axis set to 1. 
#[inline]
pub unsafe fn reduce_sum(
    x: *const f32,
    y: *mut f32,
    x_shape: [usize; 4],
    axis: usize,
    beta: f32,
) {
    let xd = x_shape;
    let mut yd = x_shape;
    yd[axis] = 1;

    for i in 0..(yd[0] * yd[1] * yd[2] * yd[3]) {
        *y.add(i) *= beta;
    }

    for n in 0..yd[0] {
        for c in 0..yd[1] {
            for h in 0..yd[2] {
                for w in 0..yd[3] {
                    let yi = n * yd[1] * yd[2] * yd[3] + c * yd[2] * yd[3] + h * yd[3] + w;
                    let yptr = y.add(yi);

                    for b in 0..xd[axis] {
                        let mut i = [n, c, h, w];
                        i[axis] = b;

                        let xi = i[0] * xd[1] * xd[2] * xd[3] + i[1] * xd[2] * xd[3] + i[2] * xd[3] + i[3];
                        *yptr += *x.add(xi);
                    }
                }
            }
        }
    }
}

/// # Reduce Sum W.r.t. X
/// - GY: Gradient w.r.t. output Y
/// - GX: Gradient w.r.t. Input X
/// - X_shape: Shape of X in the forward op.
/// - Axis: The Axis reduces in the forward op.
#[inline]
pub unsafe fn reduce_sum_wrt_x(
    gy: *const f32,
    gx: *mut f32,
    x_shape: [usize; 4],
    axis: usize,
    beta: f32,
) {
    let xd = x_shape;
    let mut yd = x_shape;
    yd[axis] = 1;

    for i in 0..(xd[0] * xd[1] * xd[2] * xd[3]) {
        *gx.add(i) *= beta;
    }

    for n in 0..yd[0] {
        for c in 0..yd[1] {
            for h in 0..yd[2] {
                for w in 0..yd[3] {
                    let yi = n * yd[1] * yd[2] * yd[3] + c * yd[2] * yd[3] + h * yd[3] + w;
                    let yptr = gy.add(yi);

                    for b in 0..xd[axis] {
                        let mut i = [n, c, h, w];
                        i[axis] = b;

                        let xi = i[0] * xd[1] * xd[2] * xd[3] + i[1] * xd[2] * xd[3] + i[2] * xd[3] + i[3];
                        *gx.add(xi) += *yptr.add(yi);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_reduce_sum() {
        let x = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let mut y = vec![0.0, 0.0, 0.0];

        unsafe {
            reduce_sum(
                x.as_ptr(),
                y.as_mut_ptr(),
                [1, 1, 3, 3],
                3,
                0.0,
            );
        }

        for i in 0..3 {
            print!("{} ", y[i])
        }
        println!("\n");

        //panic!("")
    }

    #[test]
    fn test_reduce_sum_wrt_input() {
        let mut gx = vec![0.0; 9];
        let gy = vec![6., 15., 24.];

        unsafe {
            reduce_sum_wrt_x(
                gy.as_ptr(),
                gx.as_mut_ptr(),
                [1, 1, 3, 3],
                3,
                1.0,
            )
        }

        for row in 0..3 {
            println!("");
            for col in 0..3 {
                print!("{} ", gx[row * 3 + col]);
            }
        }
        println!("");

        //panic!("")
    }
}
