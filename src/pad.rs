
/// - A: Input
/// - B: Output
/// - Dim: Dimensions of A
/// - Padh: amount to pad the height
/// - Padw: amount to pad the with
/// 
/// B has the dims (Dim0 + Padh0 + Padh1, Dim1 + Padw0 + Padw1).
/// 
/// Assumes that B is Zeros!
#[inline]
pub unsafe fn pad(
    a: *const f32,
    b: *mut f32,
    dim: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) {
    let rows = dim[0];
    let cols = dim[1];
    
    let pad_top = padh[0];
    let pad_bottom = padh[1];
    let pad_left = padw[0];
    let pad_right = padw[1];

    for i in 0..rows {
        for j in 0..cols {
            let src_index = i * cols + j;
            let dest_index = (i + pad_top) * (cols + pad_left + pad_right) + j + pad_left;
            
            let src_value = *a.offset(src_index as isize);
            *b.offset(dest_index as isize) = src_value;
        }
    }  
}

/// - GB: Gradient w.r.t. pad output
/// - GA: Gradient w.r.t. pad input
/// - Dim: Dimensions of GA
/// - Padh: amount to pad the height
/// - Padw: amount to pad the width
/// 
/// GB has the dims (Dim0 + Padh0 + Padh1, Dim1 + Padw0 + Padw1).
#[inline]
pub unsafe fn pad_wrt_a(
    gb: *const f32,
    ga: *mut f32,
    dim: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
    beta: f32,
) {
    let rows = dim[0];
    let cols = dim[1];
    let pad_top = padh[0];
    let pad_left = padw[0];
    let pad_right = padw[1];

    for i in 0..rows {
        for j in 0..cols {
            let src_index = (i + pad_top) * (cols + pad_left + pad_right) + j + pad_left;
            let dest_index = i * cols + j;

            let gradient = *gb.offset(src_index as isize);
            let gaptr = ga.offset(dest_index as isize);
            *gaptr = (*gaptr * beta) + gradient;
        }
    }
}

mod test {

    use super::*;

    #[test]
    fn test_pad() {
        // Example input matrix (3x3)
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        // Dimensions of the input matrix
        let dim = [3, 3];

        // Padding values: [top, bottom, left, right]
        let padh = [1, 1]; // Pad 1 row at the top and 1 row at the bottom
        let padw = [1, 1]; // Pad 1 column at the left and 1 column at the right

        // Calculate the dimensions of the padded matrix
        let padded_dim = [dim[0] + padh[0] + padh[1], dim[1] + padw[0] + padw[1]];

        // Create a vector to hold the padded matrix
        let mut padded_matrix: Vec<f32> = vec![0.0; padded_dim[0] * padded_dim[1]];

        unsafe {
            pad(
                input.as_ptr(),
                padded_matrix.as_mut_ptr(),
                dim,
                padh,
                padw,
            );
        }

        // Print the padded matrix
        for i in 0..padded_dim[0] {
            for j in 0..padded_dim[1] {
                let index = i * padded_dim[1] + j;
                print!("{:.1} ", padded_matrix[index]);
            }
            println!();
        }

        //panic!("")
    }

    #[test]
    fn test_pad_wrt_a() {
        // Define the dimensions of the input matrix (excluding padding)
        let input_dim = [2, 2];
    
        // Define the padding values: [top, bottom, left, right]
        let padh = [1, 1];
        let padw = [1, 1];
    
        // Create a vector to hold the gradient values (gb) for the padded output
        let gradient_padded: Vec<f32> = (0..16).map(|i| i as f32).collect();

    
        // Create a vector to hold the gradient values with respect to the input matrix (ga)
        let mut gradient_input: Vec<f32> = vec![0.0; 4]; // 2x2 matrix for input gradient
    
        unsafe {
            pad_wrt_a(
                gradient_padded.as_ptr(),
                gradient_input.as_mut_ptr(),
                input_dim,
                padh,
                padw,
                1.0,
            );
        }
    
        // Compare the calculated gradient with the expected gradient
        for i in 0..2 {
            println!("");
            for j in 0..2 {
                print!{"{} ", gradient_input[i * 2 + j]}
            }
        }
        println!("");

        //panic!("");
    }
}