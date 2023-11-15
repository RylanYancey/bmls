
use rayon::prelude::*;
use crate::error::BMLSError;
use crate::error;
use crate::Ptr;

/// # Im2col Operation
/// - X: Input
/// - Y: Output
/// - X_dim: Dimensions of X
/// - F_dim: Dimensions of Filter F
/// - Stride: h, w, strides of the filter
/// - Padh: height padding
/// - Padw: width padding
/// 
/// Y Height: fc * fh * fw
/// 
/// Y Width: (((xh - fh + (padh.0 + padh.1)) / strideh) + 1) * (((xw - fw + (padw.0 + padw.1)) / stridew) + 1) * xn
#[inline]
pub fn im2col(
    x: &[f32],
    y: &mut [f32],
    x_dim: [usize; 4],
    f_dim: [usize; 4],
    stride: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) -> Result<(), BMLSError> {
    let (nx, cx, hx, wx) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);
    let (nf, cf, hf, wf) = (f_dim[0], f_dim[1], f_dim[2], f_dim[3]);
    let (strideh, stridew) = (stride[0], stride[1]);
    let hstart = ((hx - hf + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((wx - wf + (padw[0] + padw[1])) / stridew) + 1;
    // size of the output Y
    let (ny, cy) = (hf * wf * cf, hstart * wstart * nx);

    // ensure the length of slice X is the same as its shape
    let xlen = x_dim[0]*x_dim[1]*x_dim[2]*x_dim[3];
    if x.len() != xlen {
        return error::length_mismatch("X", x.len(), "X_dim", xlen)
    }

    // ensure the length of slice Y is the same as its shape
    let ylen = ny * cy;
    if y.len() != ylen {
        return error::length_mismatch("Y", y.len(), "Y_dim", ylen);
    }

    // the kernel dimensions cannot be 0 or greater than 
    // the dimensions of the input + the padding.
    if hf == 0 || hf >= (hx+padh[0]+padh[1]) || 
       wf == 0 || wf >= (wx+padw[0]+padw[1]) ||
       cf != cx || nf == 0
    {
        return error::invalid_kernel_dim(f_dim)
    }

    // strides must not be 0
    if strideh == 0 || stridew == 0 {
        return error::invalid_strides(strideh, stridew)
    }

    let x = Ptr::new(x);
    let y = Ptr::new(y);

    (0..nx).into_par_iter().for_each(|n| {
        for h in 0..hstart {
            for w in 0..wstart {
                // the column of Y we are in
                let col = (h * wstart + w) * (n+1);
                for c in 0..cx {
                    let xrow = h * strideh;
                    let xcol = w * stridew;
                    let row = wf * hf * c;
                    let xi = n * cx * hx * wx + c * hx * wx;
                    for kw in 0..wf {
                        for kh in 0..hf {
                            let xrow = (xrow + kh) as isize - padh[0] as isize;
                            let xcol = (xcol + kw) as isize - padw[0] as isize;
                            let row = (kh * wf + kw) + row;
                            let yi = row * cy + col;
                            if xrow >= hx as isize || xrow < 0 || xcol >= wx as isize || xcol < 0 {
                                y.get_mut()[yi] = 0.0;
                                continue;
                            }
                            // the index of 
                            let xi = xi + xrow as usize * wx + xcol as usize; 
                            y.get_mut()[yi] = x.get_mut()[xi];
                        }
                    }
                }
            }
        }
    });

    Ok(())
}

/// # Im2col w.r.t. X
/// - GY: Gradient w.r.t. output Y
/// - GX: Gradient w.r.t. input X
/// - X_dim: dimensions of X in the forward op
/// - F_dim: dimensions of F in the forward op
/// - stride: H and W strides of the filter.
/// - Padh: height padding
/// - Padw: width padding
/// 
/// GY Height: fc * fh * fw
/// 
/// GY Width: (((xh - fh + (padh.0 + padh.1)) / strideh) + 1) * (((xw - fw + (padw.0 + padw.1)) / stridew) + 1) * xn
#[inline]
pub fn im2col_wrt_x(
    gy: &[f32],
    gx: &mut [f32],
    x_dim: [usize; 4],
    f_dim: [usize; 4],
    stride: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) -> Result<(), BMLSError> {
    let (nx, cx, hx, wx) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);
    let (nf, cf, hf, wf) = (f_dim[0], f_dim[1], f_dim[2], f_dim[3]);
    let (strideh, stridew) = (stride[0], stride[1]);
    let hstart = ((hx - hf + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((wx - wf + (padw[0] + padw[1])) / stridew) + 1;
    // size of the output Y
    let (ny, cy) = (hf * wf * cf, hstart * wstart * nx);

    // ensure the length of slice X is the same as its shape
    let xlen = x_dim[0]*x_dim[1]*x_dim[2]*x_dim[3];
    if gx.len() != xlen {
        return error::length_mismatch("GX", gx.len(), "X_dim", xlen)
    }

    // ensure the length of slice Y is the same as its shape
    let ylen = ny * cy;
    if gy.len() != ylen {
        return error::length_mismatch("GY", gy.len(), "GY_dim", ylen);
    }

    // the kernel dimensions cannot be 0 or greater than 
    // the dimensions of the input + the padding.
    if hf == 0 || hf >= (hx+padh[0]+padh[1]) || 
       wf == 0 || wf >= (wx+padw[0]+padw[1]) ||
       cf != cx || nf == 0
    {
        return error::invalid_kernel_dim(f_dim)
    }

    // strides must not be 0
    if strideh == 0 || stridew == 0 {
        return error::invalid_strides(strideh, stridew)
    }

    let gy = Ptr::new(gy);
    let gx = Ptr::new(gx);

    (0..nx).into_par_iter().for_each(|n| {
        for h in 0..hstart {
            for w in 0..wstart {
                // the column of Y we are in
                let col = (h * wstart + w) * (n + 1);
                for c in 0..cx {
                    let xrow = h * strideh;
                    let xcol = w * stridew;
                    // the row of Y we are in based on the channel.
                    let row = wf * hf * c;
                    let xi = n * cx * hx * wx + c * hx * wx;
                    for kh in 0..hf {
                        for kw in 0..wf {

                            let xrow = (xrow + kh) as isize - padh[0] as isize;
                            let xcol = (xcol + kw) as isize - padw[0] as isize;

                            // Adjust the row of Y we are in
                            let row = (kh * wf + kw) + row;
                            let yi = row * cy + col;

                            if xrow >= hx as isize || xrow < 0 || xcol >= wx as isize || xcol < 0 {
                                continue;
                            }
                            let xi = xi + xrow as usize * wx + xcol as usize; 
                            gx.get_mut()[xi] += gy.get_mut()[yi];
                        }
                    }
                }
            }
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn im2col() {
        // 4x4 (3 channels)
        let x: Vec<f32> = (1..=48).map(|i| i as f32).collect();
        // 12x9
        let mut y = vec![0.0; 12 * 9];

            super::im2col(
                &x,
                &mut y,
                [1, 3, 4, 4],
                [1, 3, 2, 2],
                [1, 1],
                [0, 0],
                [0, 0],
            ).unwrap();

        for row in 0..12 {
            println!("");
            for col in 0..9 {
                print!("{}, ", y[row * 9 + col])
            }
        }
        println!("");

        //panic!("");
    }
}