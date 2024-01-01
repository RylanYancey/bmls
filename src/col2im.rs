
use crate::error::BMLSError;
use crate::error;

/// # Col2Im 
/// 
/// Column Matrix (Output of Im2Col+Matmul) conversion back to image.
/// 
/// - X: C x H * W * N
/// - Y: N x C x H x W
#[inline]
pub fn col2im(
    x: &[f32],
    y: &mut [f32],
    x_dim: [usize; 2],
    y_dim: [usize; 4],
) -> Result<(), BMLSError> {
    let (xrows, xcols) = (x_dim[0], x_dim[1]);
    let (yn, yc, yh, yw) = (y_dim[0], y_dim[1], y_dim[2], y_dim[3]);

    if x.len() != y.len() {
        return error::length_mismatch("X", x.len(), "Y", y.len())
    }

    if x_dim[0] != y_dim[1] {
        return error::col2im_channel_mismatch("X", xrows, "Y", yc)
    }

    if x.len() != xrows*xcols {
        return error::length_mismatch("X", x.len(), "X_Dim", xrows*xcols)
    }

    if y.len() != yn*yc*yh*yw {
        return error::length_mismatch("Y", y.len(), "Y_Dim", yn*yc*yh*yw)
    }

    if xcols != yc*yh*yw {
        return error::length_mismatch("X_Cols", xcols, "Y - C*H*W", yc*yh*yw)
    }

    // each row is a channel
    for row in 0..xrows {
        // each col is a NxHxW
        for col in 0..xcols {
            // the batch is the column / the width of a batch (excluding the channel)
            let n = col / (yh * yw);

            // the index within HxW is wrapped by the width of a batch
            let i = col % (yh * yw);

            let h = i / yh;
            let w = i % yw;

            y[n * yc * yh * yw + row * yh * yw + h * yw + w] = x[row * xcols + col];
        }
    }

    Ok(())
}

/// # Col2Im w.r.t. X
/// - GX: C x H*W*N
/// - GY: N x C x H x W
#[inline]
pub fn col2im_wrt_x(
    gy: &[f32],
    gx: &mut [f32],
    x_dim: [usize; 2],
    y_dim: [usize; 4],
) -> Result<(), BMLSError> {
    let (xrows, xcols) = (x_dim[0], x_dim[1]);
    let (yn, yc, yh, yw) = (y_dim[0], y_dim[1], y_dim[2], y_dim[3]);

    if gy.len() != gx.len() {
        return error::length_mismatch("GY", gy.len(), "GX", gx.len())
    }

    if gy.len() != yn*yc*yh*yw {
        return error::length_mismatch("GY", gy.len(), "Y_Dim", yn*yc*yh*yw)
    } 

    if gx.len() != xrows*xcols {
        return error::length_mismatch("GX", gx.len(), "X_Dim", xrows*xcols)
    }

    if x_dim[0] != y_dim[1] {
        return error::col2im_channel_mismatch("X_chan", x_dim[0], "Y_chan", y_dim[1])
    }

    // each row is a channel
    for row in 0..xrows {
        // each col is a NxHxW
        for col in 0..xcols {
            // the batch is the column / the width of a batch (excluding the channel)
            let n = col / (yh * yw);

            // the index within HxW is wrapped by the width of a batch
            let i = col % (yh * yw);

            let h = i / yh;
            let w = i % yw;

            gx[row * xcols + col] = gy[n * yc * yh * yw + row * yh * yw + h * yw + w];
        }
    }

    Ok(())
}