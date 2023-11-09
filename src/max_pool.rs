
use rayon::prelude::*;
use super::{Ptr, PtrMut};
use crate::error::BMLSError;
use crate::error;

/// # Max Pool Operation
/// - X: Input
/// - Y: Output
/// - I: Indices of max vals in X (for backprop)
/// - X_dim: dimensions of X
/// - Kernel: HxW of the Kernel
/// - Stride: H and W strides of the Kernel
/// - Padh: Height Padding
/// - Padw: Width Padding
/// 
/// The batches and channels of Y are the same as X. \
/// The height of Y: ((xh - kh + (padh.0 + padh.1)) / strideh) + 1 \
/// The width  of Y: ((xw - kw + (padw.0 + padw.1)) / stridew) + 1
#[inline]
pub fn max_pool(
    x: &[f32],
    y: &mut [f32],
    i: &mut [usize],
    x_dim: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) -> Result<(), BMLSError> {
    let (strideh, stridew) = (stride[0], stride[1]);
    let (kernelh, kernelw) = (kernel[0], kernel[1]);
    let (xn, xc, xh, xw) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);

    let hstart = ((x_dim[2] - kernelh + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((x_dim[3] - kernelw + (padw[0] + padw[1])) / stridew) + 1;

    let (_, yc, yh, yw) = (x_dim[0], x_dim[1], hstart, wstart);

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

    // ensure the length of I is the same as Y
    if i.len() != y.len() {
        return error::length_mismatch("I", i.len(), "Y", y.len())
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
    let i = PtrMut::new(i.as_mut_ptr());

    (0..xn).into_par_iter().for_each(|n| {
        for c in 0..xc {
            for h in 0..hstart {
                for w in 0..wstart {

                    let mut max = f32::MIN;
                    let mut index = 0;

                    for kh in 0..kernelh {
                        for kw in 0..kernelw {

                            let xrow = ((h * strideh) + kh) as isize - padh[0] as isize;
                            let xcol = ((w * stridew) + kw) as isize - padw[0] as isize;

                            if xrow >= xh as isize || xrow < 0 || xcol >= xw as isize || xcol < 0 {
                                if max < 0.0 {
                                    max = 0.0;
                                    index = usize::MAX;
                                }
                                continue;
                            }

                            let xi = n * xc * xh * xw + c * xh * xw + xrow as usize * xw + xcol as usize; 
                            let val = *x.add(xi);
                            if val > max {
                                max = val;
                                index = xi;
                            }
                        }
                    }

                    let yi = n * yc * yh * yw + c * yh * yw + h * yw + w;
                    *y.add(yi) = max;
                    *i.add(yi) = index;
                }
            }
        }
    });

    Ok(())
}

/// - I: Indices of max values, returned in forward op.
/// - GY: Gradient w.r.t. output Y.
/// - GX: Gradient w.r.t. input X. 
#[inline]
pub fn max_pool_wrt_a(
    i: &[usize],
    gy: &[f32],
    gx: &mut [f32],
) -> Result<(), BMLSError> {
    if i.len() != gy.len() {
        return error::length_mismatch("I", i.len(), "GY", gy.len())
    }

    for n in 0..gy.len() {
        if i[n] != usize::MAX {
            gx[i[n]] += gy[n];
        } 
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_max_pool() {
        // Define the input matrix (4x4) for max pooling
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];

        // Create vectors to hold the pooled values and their indices
        let mut pooled_output: Vec<f32> = vec![0.0; 25];
        let mut max_indices: Vec<usize> = vec![0; 25];

            max_pool(
                &input,
                &mut pooled_output,
                &mut max_indices,
                [1, 1, 4, 4],
                [2, 2],
                [1, 1],
                [1, 1],
                [1, 1],
            ).unwrap();

        // Compare the calculated pooled values and max indices with the expected values
        for i in 0..5 {
            println!("");
            for j in 0..5 {
                print!("{} ",pooled_output[i * 5 + j]);
            }
        }
        println!("\n");

        for i in 0..5 {
            println!("");
            for j in 0..5 {
                print!("{} ", max_indices[i * 5 + j]);
            }
        }
        println!("\n");

        //panic!("")
    }

    #[test]
    fn test_max_pool_wrt_a() {

        let gb = [
            1., 2.,
            7., 4., 
        ];

        // ga is 4x4
        let mut ga = [0.0; 16];

        // 9, 3, 6, 8
        // 3, 5, 5, 4,
        // 2, 1, 7, 4,
        // 3, 9, 4, 1,

        // 2x2 indices
        let indices: [usize; 4] = [
            0, 3,
            13, 10, 
        ];

            max_pool_wrt_a(
                &indices, 
                &gb, 
                &mut ga,
            ).unwrap();

        for i in 0..4 {
            println!("");
            for j in 0..4 {
                print!("{} ", ga[i * 4 + j])
            }
        }
        println!("\n");

        //panic!("");
    }
}