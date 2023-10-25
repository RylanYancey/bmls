
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
    a: *const f32,
    b: *mut f32,
    dim: [usize; 2],
    strides: [usize; 2],
    kernel: [usize; 2],
    dilations: [usize; 2],
) {
    let input_rows = dim[0];
    let input_cols = dim[1];
    let kernel_rows = kernel[0];
    let kernel_cols = kernel[1];
    let stride_rows = strides[0];
    let stride_cols = strides[1];
    let dilation_rows = dilations[0];
    let dilation_cols = dilations[1];
    let output_rows = (input_rows - kernel_rows) / (stride_rows * dilation_rows) + 1;
    let output_cols = (input_cols - kernel_cols) / (stride_cols * dilation_cols) + 1;

    for out_row in 0..output_rows {
        for out_col in 0..output_cols {
            let patch_start_row = out_row * stride_rows;
            let patch_start_col = out_col * stride_cols;
            let patch_end_row = patch_start_row + kernel_rows;
            let patch_end_col = patch_start_col + kernel_cols;

            let mut sum = 0.0;

            for row in patch_start_row..patch_end_row {
                for col in patch_start_col..patch_end_col {

                    let dh = (row - patch_start_row) * (dilation_rows - 1);
                    let dw = (col - patch_start_col) * (dilation_cols - 1);

                    let index = (row + dh) * input_cols + (col + dw);
                    let val = *a.offset(index as isize);
                    sum += val;
                }
            }

            let output_index = out_row * output_cols + out_col;
            *b.offset(output_index as isize) = sum / (kernel_rows * kernel_cols) as f32;
        }
    }
}

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
pub unsafe fn avg_pool_wrt_a(
    gb: *const f32,
    ga: *mut f32,
    dim: [usize; 2],
    strides: [usize; 2],
    kernel: [usize; 2],
    dilations: [usize; 2],
) {
    let rows = dim[0];
    let cols = dim[1];
    let kernel_rows = kernel[0];
    let kernel_cols = kernel[1];
    let stride_rows = strides[0];
    let stride_cols = strides[1];
    let dilation_rows = dilations[0];
    let dilation_cols = dilations[1];
    let output_rows = (rows - kernel_rows) / (stride_rows * dilation_rows) + 1;
    let output_cols = (cols - kernel_cols) / (stride_cols * dilation_cols) + 1;

    for out_row in 0..output_rows {
        for out_col in 0..output_cols {
            let patch_start_row = out_row * stride_rows;
            let patch_start_col = out_col * stride_cols;
            let patch_end_row = patch_start_row + kernel_rows;
            let patch_end_col = patch_start_col + kernel_cols;

            let gradient = *gb.offset((out_row * output_cols + out_col) as isize);
            let average = gradient / (kernel_rows * kernel_cols) as f32;

            for row in patch_start_row..patch_end_row {
                for col in patch_start_col..patch_end_col {

                    let dh = (row - patch_start_row) * (dilation_rows - 1);
                    let dw = (col - patch_start_col) * (dilation_cols - 1);

                    let index = (row + dh) * cols + (col + dw);
                    *ga.offset(index as isize) += average;
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
    
        // Dimensions of the input matrix
        let dim = [4, 4];
    
        // Pooling parameters
        let strides = [1, 1];
        let kernel = [2, 2];
        let dilations = [2, 2];
    
        // Create a vector to hold the pooled values
        let mut pooled_output: Vec<f32> = vec![0.0; 4];
    
        unsafe {
            avg_pool(
                input.as_ptr(),
                pooled_output.as_mut_ptr(),
                dim,
                strides,
                kernel,
                dilations,
            );
        }
    
        for i in 0..2 {
            println!("");
            for j in 0..2 {
                print!("{} ", pooled_output[i * 2 + j]);
            }
        }
        println!("");

        //panic!("");
    } 

    #[test]
    fn test_avg_pool_wrt_a() {
        // Dimensions of the input matrix a (4x4)
        let input_dim = [4, 4];
    
        // Pooling parameters
        let strides = [2, 2];
        let kernel = [2, 2];
        let dilations = [1, 1];
    
        // Create a vector to hold the gradient values (gb) for the output
        let gradient_output: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    
        // Create a vector to hold the gradient with respect to the input (ga)
        let mut gradient_input: Vec<f32> = vec![0.0; 16]; // 4x4 input matrix
    
        unsafe {
            avg_pool_wrt_a(
                gradient_output.as_ptr(),
                gradient_input.as_mut_ptr(),
                input_dim,
                strides,
                kernel,
                dilations,
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