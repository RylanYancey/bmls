
/// # Avg Pooling Operation
/// - A: Input
/// - B: Output
/// - Dim: Dimensions of A. 
/// - Strides: Distance between patches
/// - Kernel: Size of the Kernel
/// - Dilations: Amount to dilate. Cannot be Zero. 
/// 
/// B Should have the height: (input_rows - kernel_rows) / (stride_rows * dilation_rows) + 1
/// B should have the width: (input_cols - kernel_cols) / (stride_cols * dilation_cols) + 1
#[inline]
pub unsafe fn avg_pool(
    x: *const f32,
    y: *mut f32,
    x_dim: [usize; 4],
    stride: [usize; 2],
    kernel: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) {
    let (strideh, stridew) = (stride[0], stride[1]);
    let (kernelh, kernelw) = (kernel[0], kernel[1]);
    let (xn, xc, xh, xw) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);

    let hstart = ((x_dim[2] - kernelh + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((x_dim[3] - kernelw + (padw[0] + padw[1])) / stridew) + 1;

    let (_, yc, yh, yw) = (x_dim[0], x_dim[1], hstart, wstart);

    let k_len = kernelh * kernelw;

    for n in 0..xn {
        for c in 0..xc {
            for h in 0..hstart {
                for w in 0..wstart {

                    let mut sum = 0.0;

                    for kh in 0..kernelh {
                        for kw in 0..kernelw {

                            let xrow = ((h * strideh) + kh) as isize - padh[0] as isize;
                            let xcol = ((w * stridew) + kw) as isize - padw[0] as isize;

                            if xrow >= xh as isize || xrow < 0 || xcol >= xw as isize || xcol < 0 {
                                continue;
                            }

                            let xi = n * xc * xh * xw + c * xh * xw + xrow as usize * xw + xcol as usize; 
                            sum += *x.add(xi);
                        }
                    }
                    let yi = n * yc * yh * yw + c * yh * yw + h * yw + w;
                    *y.add(yi) = sum / k_len as f32;
                }
            }
        }
    }
}

/// Avg Pooling w.r.t. X
/// - GB: Output Gradient
/// - GA: Input Gradient
/// - Dim: Dimensions of A
/// - Strides: Distance between patches
/// - Kernel: Size of the Kernel
/// - Dilations: Amount to dilate. Cannot be Zero.
/// 
/// B Should have the height: (input_rows - kernel_rows) / (stride_rows * dilation_rows) + 1
/// B should have the width: (input_cols - kernel_cols) / (stride_cols * dilation_cols) + 1
/// 
/// GA is expected to be zeroed. 
#[inline]
pub unsafe fn avg_pool_wrt_x(
    gy: *const f32,
    gx: *mut f32,
    x_dim: [usize; 4],
    stride: [usize; 2],
    kernel: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) {
    let (strideh, stridew) = (stride[0], stride[1]);
    let (kernelh, kernelw) = (kernel[0], kernel[1]);
    let (xn, xc, xh, xw) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);

    let hstart = ((x_dim[2] - kernelh + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((x_dim[3] - kernelw + (padw[0] + padw[1])) / stridew) + 1;

    let (_, yc, yh, yw) = (x_dim[0], x_dim[1], hstart, wstart);

    let k_len = kernelh * kernelw;

    for n in 0..xn {
        for c in 0..xc {
            for h in 0..hstart {
                for w in 0..wstart {

                    let yi = n * yc * yh * yw + c * yh * yw + h * yw + w;

                    for kh in 0..kernelh {
                        for kw in 0..kernelw {

                            let xrow = ((h * strideh) + kh) as isize - padh[0] as isize;
                            let xcol = ((w * stridew) + kw) as isize - padw[0] as isize;

                            if xrow >= xh as isize || xrow < 0 || xcol >= xw as isize || xcol < 0 {
                                continue;
                            }

                            let xi = n * xc * xh * xw + c * xh * xw + xrow as usize * xw + xcol as usize; 
                            *gx.add(xi) += *gy.add(yi) / k_len as f32;
                        }
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
    
        unsafe {
            avg_pool(
                input.as_ptr(),
                pooled_output.as_mut_ptr(),
                [1, 1, 4, 4],
                [1, 1],
                [2, 2],
                [1, 1],
                [1, 1],
            );
        }
    
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
    
        unsafe {
            avg_pool_wrt_x(
                gradient_output.as_ptr(),
                gradient_input.as_mut_ptr(),
                [1, 1, 4, 4],
                [2, 2],
                [1, 1],
                [0, 0],
                [0, 0],
            );
        }
    
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