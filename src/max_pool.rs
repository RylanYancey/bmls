
/// - A: Input
/// - B: Output
/// - I: Index of max value in a patch
/// - Dim: Dimensions of A. 
/// - Strides: Distance between patches
/// - Kernel: Size of the Kernel
/// - Dilations: Amount to dilate. Cannot be Zero. 
/// 
/// B Should have the height: (input_rows - kernel_rows) / (stride_rows * dilation_rows) + 1
/// B should have the width: (input_cols - kernel_cols) / (stride_cols * dilation_cols) + 1
#[inline]
pub unsafe fn max_pool(
    a: *const f32,
    b: *mut f32,
    i: *mut usize,
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

            let mut max_val = std::f32::NEG_INFINITY;
            let mut max_idx = 0;

            for row in patch_start_row..patch_end_row {
                for col in patch_start_col..patch_end_col {

                    let dh = (row - patch_start_row) * (dilation_rows - 1);
                    let dw = (col - patch_start_col) * (dilation_cols - 1);

                    let index = (row + dh) * input_cols + (col + dw);
                    let val = *a.offset(index as isize);

                    if val > max_val {
                        max_val = val;
                        max_idx = index;
                    }
                }
            }

            let output_index = out_row * output_cols + out_col;
            *b.offset(output_index as isize) = max_val;
            *i.offset(output_index as isize) = max_idx;
        }
    }
}

/// - I: Indices of max values, returned in forward op.
/// - GB: Gradient w.r.t. output B.
/// - GA: Gradient w.r.t. input A. 
/// - Dim: Dimensions of GB. 
#[inline]
pub unsafe fn max_pool_wrt_a(
    i: *const usize,
    gb: *const f32,
    ga: *mut f32,
    dim: [usize; 2],
) {
    let len = dim[0] * dim[1];

    for n in 0..len {
        *ga.add(*i.add(n)) += *gb.add(n); 
    }
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

        // Dimensions of the input matrix
        let dim = [4, 4];

        // Pooling parameters
        let strides = [2, 2];
        let kernel = [2, 2];
        let dilations = [1, 1];

        // Create vectors to hold the pooled values and their indices
        let mut pooled_output: Vec<f32> = vec![0.0; 4];
        let mut max_indices: Vec<usize> = vec![0; 4];

        unsafe {
            max_pool(
                input.as_ptr(),
                pooled_output.as_mut_ptr(),
                max_indices.as_mut_ptr(),
                dim,
                strides,
                kernel,
                dilations,
            );
        }

        // Compare the calculated pooled values and max indices with the expected values
        for i in 0..2 {
            println!("");
            for j in 0..2 {
                print!("{} ",pooled_output[i * 2 + j]);
            }
        }
        println!("\n");

        for i in 0..2 {
            println!("");
            for j in 0..2 {
                print!("{} ", max_indices[i * 2 + j]);
            }
        }
        println!("\n");

        //panic!("")
    }

    #[test]
    fn test_max_pool_wrt_a() {
        let dim_gb = [2, 2];

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

        unsafe {
            max_pool_wrt_a(
                indices.as_ptr(), 
                gb.as_ptr(), 
                ga.as_mut_ptr(), 
                dim_gb
            )
        }

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