
use crate::error::BMLSError;
use crate::error;

/// # Reduce Mean Operator
/// - X: Input
/// - Y: Output
/// - X_shape: shape of X
/// - Axis: Axis to sum
/// 
/// Y Shape is the same as X, but with the specified Axis set to 1. 
#[inline]
pub fn reduce_mean(
    x: &[f32],
    y: &mut [f32],
    x_shape: [usize; 4],
    axis: usize,
) -> Result<(), BMLSError> {
    let xd = x_shape;
    let mut yd = x_shape;
    yd[axis] = 1; 

    let ylen = yd[0]*yd[1]*yd[2]*yd[3];
    if y.len() != ylen {
        return error::length_mismatch("Y", y.len(), "YDim", ylen);
    }

    let xlen = xd[0]*xd[1]*xd[2]*xd[3];
    if x.len() != xlen {
        return error::length_mismatch("X", x.len(), "XDim", xlen);
    }

    for n in 0..yd[0] {
        for c in 0..yd[1] {
            for h in 0..yd[2] {
                for w in 0..yd[3] {
                    let yi = n * yd[1] * yd[2] * yd[3] + c * yd[2] * yd[3] + h * yd[3] + w;
                    for b in 0..xd[axis] {
                        let mut i = [n, c, h, w];
                        i[axis] = b;

                        let xi = i[0] * xd[1] * xd[2] * xd[3] + i[1] * xd[2] * xd[3] + i[2] * xd[3] + i[3];
                        y[yi] += x[xi]
                    }

                    y[yi] /= xd[axis] as f32;
                }
            }
        }
    }

    Ok(())
}

/// # Reduce Mean W.r.t. X
/// - GY: Gradient w.r.t. Output Y
/// - GX: Gradient w.r.t. Input X
/// - X_Shape: Shape of X in the forward op.
/// - Axis: Axis to be reduced
#[inline]
pub fn reduce_mean_wrt_x(
    gy: &[f32],
    gx: &mut [f32],
    x_shape: [usize; 4],
    axis: usize,
) -> Result<(), BMLSError> {
    let xd = x_shape;
    let mut yd = x_shape;
    yd[axis] = 1;

    let ylen = yd[0]*yd[1]*yd[2]*yd[3];
    if gy.len() != ylen {
        return error::length_mismatch("GY", gy.len(), "GYDim", ylen);
    }

    let xlen = xd[0]*xd[1]*xd[2]*xd[3];
    if gx.len() != xlen {
        return error::length_mismatch("GX", gx.len(), "GXDim", xlen);
    }

    let len = xd[axis] as f32;
    for n in 0..yd[0] {
        for c in 0..yd[1] {
            for h in 0..yd[2] {
                for w in 0..yd[3] {
                    let yi = n * yd[1] * yd[2] * yd[3] + c * yd[2] * yd[3] + h * yd[3] + w;
                    let val = gy[yi] / len;

                    for b in 0..xd[axis] {
                        let mut i = [n, c, h, w];
                        i[axis] = b;

                        let xi = i[0] * xd[1] * xd[2] * xd[3] + i[1] * xd[2] * xd[3] + i[2] * xd[3] + i[3];
                        gx[xi] += val; 
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_mean() {
        let x = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let mut y = vec![0.0; 3];

            reduce_mean(
                &x,
                &mut y,
                [1, 1, 3, 3],
                2,
            ).unwrap();

        for i in 0..3 {
            print!("{} ", y[i]);
        }
        println!("");

        //panic!("")
    }

    #[test]
    fn test_reduce_mean_wrt_x() {
        let mut gx = vec![0.0; 9];
        let gy = vec![4.0, 5.0, 9.0];

            reduce_mean_wrt_x(
                &gy,
                &mut gx,
                [1, 1, 3, 3],
                2, 
            ).unwrap();

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