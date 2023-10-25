
#[inline]
pub unsafe fn softmax(
    a: *const f32,
    b: *mut f32,
    dim: [usize; 2]
) {
    let rows = dim[0];
    let cols = dim[1];


    for i in 0..rows {
        let a_row = a.offset((i * cols) as isize);
        let b_row = b.offset((i * cols) as isize);
        
        // Find the maximum value in the row
        let mut max_val = *a_row;
        for j in 1..cols {
            let val = *a_row.offset(j as isize);
            if val > max_val {
                max_val = val;
            }
        }
        
        // Compute the softmax for each element in the row
        let mut sum_exp = 0.0;
        for j in 0..cols {
            let val = *a_row.offset(j as isize) - max_val; // Subtract max for numerical stability
            let exp_val = val.exp();
            *b_row.offset(j as isize) = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize the row by dividing by the sum of exponentials
        for j in 0..cols {
            *b_row.offset(j as isize) /= sum_exp;
        }
    }
}

#[inline]
pub unsafe fn softmax_wrt_a(
    b: *const f32,
    gb: *const f32,
    ga: *mut f32,
    dim: [usize; 2],
    beta: f32,
) {
    let rows = dim[0];
    let cols = dim[1];

    for i in 0..rows {
        let b_row = b.offset((i * cols) as isize);
        let gb_row = gb.offset((i * cols) as isize);
        let ga_row = ga.offset((i * cols) as isize);
        
        for j in 0..cols {
            let softmax_val = *b_row.offset(j as isize);
            let gradient_b = *gb_row.offset(j as isize);
            
            let mut sum = 0.0;
            for k in 0..cols {
                let other_softmax_val = *b_row.offset(k as isize);
                
                if j == k {
                    sum += gradient_b * (softmax_val * (1.0 - softmax_val));
                } else {
                    sum += gradient_b * (-other_softmax_val * softmax_val);
                }
            }
            
            let gaij = ga_row.offset(j as isize);
            *gaij = (*gaij * beta) + sum;
        }
    }
}

#[cfg(test)]
mod tests {
    
    use super::*;

    #[test]
    fn test_softmax() {
        let input_data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output_data: [f32; 6] = [0.0; 6];
        let dim = [2, 3]; // Example dimensions for a 2x3 matrix
    
        unsafe {
            softmax(input_data.as_ptr(), output_data.as_mut_ptr(), dim);
        }
    
        // Print the result
        for i in 0..dim[0] {
            println!("");
            for j in 0..dim[1] {
                let idx = i * dim[1] + j;
                println!("Input: {}, Softmax Output: {}", input_data[idx], output_data[idx]);
            }
        }

        //panic!("");
    }

    #[test]
    fn test_wrt_a() {
        let softmax_output: [f32; 6] = [0.1, 0.3, 0.6, 0.2, 0.4, 0.4];
        let gradient_b: [f32; 6] = [0.2, 0.3, 0.1, 0.4, 0.2, 0.3];
        let mut gradient_a: [f32; 6] = [0.0; 6];
        let dim = [2, 3]; // Example dimensions for a 2x3 matrix
    
        unsafe {
            softmax_wrt_a(
                softmax_output.as_ptr(),
                gradient_b.as_ptr(),
                gradient_a.as_mut_ptr(),
                dim,
                0.0,
            );
        }
    
        // Print the result
        for i in 0..dim[0] {
            for j in 0..dim[1] {
                let idx = i * dim[1] + j;
                println!(
                    "Softmax Output: {}, Gradient_b: {}, Gradient_a: {}",
                    softmax_output[idx],
                    gradient_b[idx],
                    gradient_a[idx]
                );
            }
        }

        //panic!("")
    }
}