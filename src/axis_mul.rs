
use crate::error::BMLSError;
use crate::error;

#[inline]
pub fn axis_mul(
    x1: &[f32],
    x2: &[f32],
    y: &mut [f32],
    dim: [usize; 4],
    axis: usize,
) -> Result<(), BMLSError> {
    // the expected lengths of X1 and Y.
    let len = dim[0]*dim[1]*dim[2]*dim[3];

    if y.len() != len {
        return error::length_mismatch("Y", y.len(), "Dim", len)
    }

    if x1.len() != y.len() {
        return error::length_mismatch("X1", x1.len(), "Y", y.len())
    }

    if x2.len() != dim[axis] {
        return error::axis_mismatch(0, "X2", x2.len(), axis, "Y", dim[axis])
    }

    for n in 0..dim[0] {
        for c in 0..dim[1] {
            for h in 0..dim[2] {
                for w in 0..dim[3] {
                    let i = n * dim[1] * dim[2] * dim[3] + c * dim[2] * dim[3] + h * dim[3] + w;
                    let indices = [n, c, h, w];

                    y[i] = x1[i] * x2[indices[axis]]
                }
            }
        }
    }

    Ok(())
}

#[inline]
pub unsafe fn axis_mul_wrt_x1(
    x2: &[f32],
    gy: &[f32],
    g1: &mut [f32],
    dim: [usize; 4],
    axis: usize,
) -> Result<(), BMLSError> {
    // the expected lengths of X1 and Y.
    let len = dim[0]*dim[1]*dim[2]*dim[3];

    if gy.len() != len {
        return error::length_mismatch("GY", gy.len(), "Dim", len)
    }

    if x2.len() != dim[axis] {
        return error::axis_mismatch(0, "X2", x2.len(), axis, "Y", dim[axis])
    }

    if g1.len() != gy.len() {
        return error::length_mismatch("G1", g1.len(), "GY", gy.len())
    }

    for n in 0..dim[0] {
        for c in 0..dim[1] {
            for h in 0..dim[2] {
                for w in 0..dim[3] {
                    let i = n * dim[1] * dim[2] * dim[3] + c * dim[2] * dim[3] + h * dim[3] + w;
                    let indices = [n, c, h, w];

                    g1[i] += gy[i] * x2[indices[axis]]
                }
            }
        }
    }

    Ok(())
}


#[inline]
pub unsafe fn axis_mul_wrt_x2(
    x1: &[f32],
    gy: &[f32],
    g2: &mut [f32],
    dim: [usize; 4],
    axis: usize,
) -> Result<(), BMLSError> {
    // the expected lengths of X1 and Y.
    let len = dim[0]*dim[1]*dim[2]*dim[3];

    if gy.len() != len {
        return error::length_mismatch("GY", gy.len(), "Dim", len)
    }

    if x1.len() != gy.len() {
        return error::length_mismatch("X1", x1.len(), "GY", gy.len())
    }

    if g2.len() != dim[axis] {
        return error::axis_mismatch(0, "G2", g2.len(), axis, "GY", dim[axis])
    }

    for n in 0..dim[0] {
        for c in 0..dim[1] {
            for h in 0..dim[2] {
                for w in 0..dim[3] {
                    let i = n * dim[1] * dim[2] * dim[3] + c * dim[2] * dim[3] + h * dim[3] + w;
                    let indices = [n, c, h, w];

                    g2[indices[axis]] += gy[i] * x1[i]
                }
            }
        }
    }

    Ok(())
}
