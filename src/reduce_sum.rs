
pub fn reduce_sum(
    a: *const f32,
    b: *mut f32,
    a_shape: [usize; 4],
    b_shape: [usize; 4],
    axes: &[usize],
) {
    let mut a_strides: [isize; 4] = [0; 4];
    let mut b_strides: [isize; 4] = [0; 4];
    let mut keep_dims: [bool; 4] = [false; 4];

    let num_axes = axes.len();
    let num_dims = 4;

    // Calculate strides for both input and output tensors
    let mut stride = 1;
    for i in (0..num_dims).rev() {
        a_strides[i] = stride;
        b_strides[i] = stride;
        stride *= a_shape[i] as isize;
    }

    // Set keep_dims to true for axes to be reduced
    for axis in axes {
        keep_dims[*axis] = true;
    }

    // Iterate over elements in input tensor
    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            for k in 0..a_shape[2] {
                for l in 0..a_shape[3] {
                    // Check if this element is in the reduced axes
                    let mut reduced = false;
                    for axis in axes {
                        if keep_dims[*axis] {
                            reduced = true;
                            break;
                        }
                    }

                    if !reduced {
                        // Calculate the corresponding position in the output tensor
                        let mut b_index: isize = 0;
                        for axis in axes {
                            b_index += *axis as isize * b_strides[*axis];
                        }
                        b_index += i as isize * b_strides[0]
                            + j as isize * b_strides[1]
                            + k as isize * b_strides[2]
                            + l as isize * b_strides[3];

                        // Sum the element along the reduced axes
                        unsafe {
                            *b.offset(b_index) += *a.offset(
                                i as isize * a_strides[0]
                                    + j as isize * a_strides[1]
                                    + k as isize * a_strides[2]
                                    + l as isize * a_strides[3],
                            );
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
    fn test_reduce_sum() {
        let a_shape = [2, 3, 4, 5];
        let b_shape = [2, 3, 1, 5];
        let axes = vec![2];
    
        // Create example input tensor A (flatten for raw pointer)
        let mut a_data: Vec<f32> = (0..2 * 3 * 4 * 5).map(|x| x as f32).collect();
    
        // Allocate memory for the output tensor B
        let mut b_data: Vec<f32> = vec![0.0; 2 * 3 * 1 * 5];
    
        // Call the reduce_sum function
        reduce_sum(
            a_data.as_ptr(),
            b_data.as_mut_ptr(),
            a_shape,
            b_shape,
            &axes,
        );
    
        // Print the output tensor B
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..1 {
                    for l in 0..5 {
                        let index = i * 3 * 1 * 5 + j * 1 * 5 + k * 5 + l;
                        print!("{:.2} ", b_data[index]);
                    }
                    println!();
                }
            }
        }

        //panic!("");
    }

    #[test]
    fn test_reduce_sum_wrt_input() {

    }
}
