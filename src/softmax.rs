
use crate::error::BMLSError;
use crate::error;

#[inline]
pub fn softmax(
    x: &[f32],
    y: &mut [f32],
    dim: [usize; 2],
) -> Result<(), BMLSError> {
    let rows = dim[0];
    let cols = dim[1];

    let len = rows * cols;
    if x.len() != len {
        return error::length_mismatch("X", x.len(), "Dim", len);
    }

    if y.len() != len { 
        return error::length_mismatch("Y", y.len(), "Dim", len);
    }

    for i in 0..rows {
        // the max value in the row.
        // we will take e^x - max for stability.
        // as input values become larger, they converge to NaN,
        // however, negative numbers converge to 0. 
        // by subtracting the max, whatever the max value is will
        // output as 1, and everything else as 0, resulting in a 
        // cleaner and more stable output.
        // let mut max = x[i * cols];
        // for j in 1..cols {
        //     if x[i * cols + j] > max {
        //         max = x[i * cols + j];
        //     }
        // }

        // calculate the sum and assign e^x-max to y.
        let mut sum = 0.0;
        for j in 0..cols {
            // y = e^(x+max) (later on we will divide by sum)
            y[i * cols + j] = f32::exp(x[i * cols + j]);
            sum += y[i * cols + j];
        }

        // divide the y value by the sum.
        for j in 0..cols {
            y[i * cols + j] /= sum;
        }
    }

    Ok(())
}

#[inline]
pub fn softmax_wrt_x(
    y: &[f32],
    gy: &[f32],
    g1: &mut [f32],
    dim: [usize; 2],
) -> Result<(), BMLSError> {
    let rows = dim[0];
    let cols = dim[1];

    let len = rows * cols;
    if y.len() != len {
        return error::length_mismatch("Y", y.len(), "Dim", len);
    }

    if gy.len() != len {
        return error::length_mismatch("GY", gy.len(), "Dim", len);
    }

    if g1.len() != len {
        return error::length_mismatch("G1", g1.len(), "Dim", len);
    }

    for i in 0..rows {
        for j in 0..cols {
            let v = y[i * cols + j];

            let mut sum = 0.0;
            for k in 0..cols {
                if i == j {
                    // when i and j are the same, gx is
                    // gy * (y * (1. - y));
                    sum += gy[i * cols + k] * (v * (1. - v));
                } else {
                    // when i and j are not the same, gx is
                    // gy * -other_y*y
                    sum += gy[i * cols + k] * (-y[i * cols + k] * v);
                }
            }

            g1[i * cols + j] += sum;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    
    use super::*;

    #[test]
    fn test_softmax() {
        let input_data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output_data: [f32; 6] = [0.0; 6];
        let dim = [2, 3]; // Example dimensions for a 2x3 matrix
    
        softmax(&input_data, &mut output_data, dim).unwrap();
    
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
    
            softmax_wrt_x(
                &softmax_output,
                &gradient_b,
                &mut gradient_a,
                dim,
            ).unwrap();
    
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