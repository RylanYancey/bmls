
use rayon::prelude::*;
use super::{Ptr, PtrMut};
use crate::error::BMLSError;
use crate::error;

/// # Avg Pooling Operation
/// - A: Input
/// - B: Output
/// - Dim: Dimensions of A. 
/// - Strides: Distance between patches
/// - Kernel: Size of the Kernel
/// - Dilations: Amount to dilate. Cannot be Zero. 
/// 
/// B Should have the height: ((input_rows - kernel_rows + (padh0 + padh1)) / stride_rows) + 1
/// B should have the width: ((input_cols - kernel_cols + (padw0 + padw1)) / stride_cols) + 1
#[inline]
pub fn avg_pool(
    x: &[f32],
    y: &mut [f32],
    x_dim: [usize; 4],
    stride: [usize; 2],
    kernel: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) -> Result<(), BMLSError> {
    let (strideh, stridew) = (stride[0], stride[1]);
    let (kernelh, kernelw) = (kernel[0], kernel[1]);
    let (xn, xc, xh, xw) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);

    let hstart = ((x_dim[2] - kernelh + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((x_dim[3] - kernelw + (padw[0] + padw[1])) / stridew) + 1;

    let (_, yc, yh, yw) = (x_dim[0], x_dim[1], hstart, wstart);

    let k_len = kernelh * kernelw;

    // ensure the length of slice X is the same as its shape
    let xlen = x_dim[0]*x_dim[1]*x_dim[2]*x_dim[3];
    if x.len() != xlen {
        return error::length_mismatch("X", x.len(), "X_dim", xlen)
    }

    // ensure the length of slice Y is the same as its shape
    let ylen = xn * yc * yh * yw;
    if y.len() != ylen {
        return error::length_mismatch("Y", y.len(), "Y_dim", ylen);
    }

    // the kernel dimensions cannot be 0 or greater than 
    // the dimensions of the input + the padding.
    if kernel[0] == 0 || kernel[0] >= (xh+padh[0]+padh[1]) || 
       kernel[1] == 0 || kernel[1] >= (xw+padw[0]+padw[1]) 
    {
        return error::invalid_kernel_dim([1, 1, kernelh, kernelw])
    }

    // strides must not be 0
    if strideh == 0 || stridew == 0 {
        return error::invalid_strides(strideh, stridew)
    }

    let x = Ptr::new(x.as_ptr());
    let y = PtrMut::new(y.as_mut_ptr());

    (0..xn).into_par_iter().for_each(|n| {
        for c in 0..xc {
            for h in 0..hstart {
                let xrow = h * strideh;
                let xi = n * xc * xh * xw + c * xh * xw;
                for w in 0..wstart {
                    let yi = n * yc * yh * yw + c * yh * yw + h * yw + w;
                    let xcol = w * stridew;
                    let mut sum = 0.0;
                    for kh in 0..kernelh {
                        let xrow = (xrow + kh) as isize - padh[0] as isize;
                        for kw in 0..kernelw {
                            let xcol = (xcol + kw) as isize - padw[0] as isize;
                            if xrow >= xh as isize || xrow < 0 || xcol >= xw as isize || xcol < 0 {
                                continue;
                            }
                            let xi = xi + xrow as usize * xw + xcol as usize; 
                            sum += *x.add(xi);
                        }
                    }
                    *y.add(yi) = sum / k_len as f32;
                }
            }
        }
    });

    Ok(())
}

/// Avg Pooling w.r.t. X
/// - GY: Output Gradient
/// - GX: Input Gradient
/// - XDim: Dimensions of X
/// - Strides: Distance between patches
/// - Kernel: Size of the Kernel
/// 
/// GY Should have the height: ((input_rows - kernel_rows + (padh0 + padh1)) / stride_rows) + 1
/// GY should have the width: ((input_cols - kernel_cols + (padw0 + padw1)) / stride_cols) + 1
#[inline]
pub fn avg_pool_wrt_x(
    gy: &[f32],
    gx: &mut [f32],
    x_dim: [usize; 4],
    stride: [usize; 2],
    kernel: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) -> Result<(), BMLSError> {
    let (strideh, stridew) = (stride[0], stride[1]);
    let (kernelh, kernelw) = (kernel[0], kernel[1]);
    let (xn, xc, xh, xw) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);

    let hstart = ((x_dim[2] - kernelh + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((x_dim[3] - kernelw + (padw[0] + padw[1])) / stridew) + 1;

    let (_, yc, yh, yw) = (x_dim[0], x_dim[1], hstart, wstart);

    let k_len = kernelh * kernelw;

    // ensure the length of slice X is the same as its shape
    let xlen = x_dim[0]*x_dim[1]*x_dim[2]*x_dim[3];
    if gx.len() != xlen {
        return error::length_mismatch("GX", gx.len(), "GX_dim", xlen)
    }

    // ensure the length of slice Y is the same as its shape
    let ylen = xn * yc * yh * yw;
    if gy.len() != ylen {
        return error::length_mismatch("GY", gy.len(), "GY_dim", ylen);
    }

    // the kernel dimensions cannot be 0 or greater than 
    // the dimensions of the input + the padding.
    if kernel[0] == 0 || kernel[0] >= (xh+padh[0]+padh[1]) || 
       kernel[1] == 0 || kernel[1] >= (xw+padw[0]+padw[1]) 
    {
        return error::invalid_kernel_dim([1, 1, kernelh, kernelw])
    }

    // strides must not be 0
    if strideh == 0 || stridew == 0 {
        return error::invalid_strides(strideh, stridew)
    }

    let gy = Ptr::new(gy.as_ptr());
    let gx = PtrMut::new(gx.as_mut_ptr());

    (0..xn).into_par_iter().for_each(|n| {
        for c in 0..xc {
            for h in 0..hstart {
                let xrow = h * strideh;
                let xi = n * xc * xh * xw + c * xh * xw;
                for w in 0..wstart {
                    let yi = n * yc * yh * yw + c * yh * yw + h * yw + w;
                    let xcol = w * stridew;
                    for kh in 0..kernelh {
                        let xrow = (xrow + kh) as isize - padh[0] as isize;
                        for kw in 0..kernelw {
                            let xcol = (xcol + kw) as isize - padw[0] as isize;
                            if xrow >= xh as isize || xrow < 0 || xcol >= xw as isize || xcol < 0 {
                                continue;
                            }
                            let xi = xi + xrow as usize * xw + xcol as usize; 
                            *gx.add(xi) += *gy.add(yi) / k_len as f32;
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
    
    use super::*;

    #[test]
    fn test_avg_pool() {
        // Define the input matrix (4x4) for average pooling
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
    
        // Create a vector to hold the pooled values
        let mut pooled_output: Vec<f32> = vec![0.0; 25];
    
            avg_pool(
                &input,
                &mut pooled_output,
                [1, 1, 4, 4],
                [1, 1],
                [2, 2],
                [1, 1],
                [1, 1],
            ).unwrap();
    
        for i in 0..5 {
            println!("");
            for j in 0..5 {
                print!("{} ", pooled_output[i * 5 + j]);
            }
        }
        println!("");

        //panic!("");
    } 

    #[test]
    fn test_avg_pool_wrt_a() {
        // Create a vector to hold the gradient values (gb) for the output
        let gradient_output: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    
        // Create a vector to hold the gradient with respect to the input (ga)
        let mut gradient_input: Vec<f32> = vec![1.0; 16]; // 4x4 input matrix
    
            avg_pool_wrt_x(
                &gradient_output,
                &mut gradient_input,
                [1, 1, 4, 4],
                [2, 2],
                [1, 1],
                [0, 0],
                [0, 0],
            ).unwrap();
    
        for i in 0..4 {
            println!("");
            for j in 0..4 {
                print!("{} ", gradient_input[i * 4 + j]);
            }
        }
        println!("\n");

        //panic!("");
    }
}