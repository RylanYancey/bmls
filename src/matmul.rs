
use matrixmultiply::sgemm;

#[inline]
pub unsafe fn matmul(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    a_dim: [usize; 2],
    b_dim: [usize; 2],
    beta: f32,
) {
    sgemm(
        a_dim[0],                 // M: Rows of A
        a_dim[1],                 // K: Columns of A
        b_dim[1],                 // N: Columns of B
        1.0,                      // Alpha (scaling factor for A*B)
        a,
        a_dim[1] as isize,        // Leading dimension of A
        1,                        // Column stride of A
        b,
        b_dim[1] as isize,        // Leading dimension of B
        1,                        // Column stride of B
        beta,                     // Beta (scaling factor for C)
        c,
        b_dim[1] as isize,        // Leading dimension of C
        1,                        // Column stride of C
    );
}

#[inline]
pub unsafe fn matmul_wrt_a(
    gc: *const f32,
    b: *const f32,
    ga: *mut f32,
    a_dim: [usize; 2],
    b_dim: [usize; 2],
    beta: f32,
) {
    sgemm(
        a_dim[0],                    // M: Rows of A
        b_dim[1],                    // K: Columns of A
        b_dim[0],                    // N: Rows of B (transposed)
        1.0,                         // Alpha (scaling factor for A*B)
        gc,                          // Input gradient for C
        b_dim[1] as isize,           // Leading dimension of gc
        1,                           // Column stride of gc
        b,                           // Matrix B
        b_dim[0] as isize,           // Leading dimension of B (transposed)
        1,                           // Column stride of B
        beta,                         // Beta (scaling factor for ga)
        ga,                          // Output gradient for A
        a_dim[1] as isize,           // Leading dimension of ga
        1,                           // Column stride of ga
    );
}

#[inline]
pub unsafe fn matmul_wrt_b(
    a: *const f32,
    gc: *const f32,
    gb: *mut f32,
    a_dim: [usize; 2],
    b_dim: [usize; 2],
    beta: f32,
) {
    // Check that the dimensions are compatible for multiplication
    assert_eq!(a_dim[1], b_dim[0]);

    // Compute the gradient of the matrix product
    sgemm(
        a_dim[0],                    // M: Rows of A
        a_dim[1],                    // K: Columns of A
        b_dim[1],                    // N: Columns of B
        1.0,                         // Alpha (scaling factor for A*B)
        a,                           // Matrix A
        a_dim[0] as isize,           // Leading dimension of A
        1,                           // Column stride of A
        gc,                          // Input gradient for C
        a_dim[1] as isize,           // Leading dimension of gc
        1,                           // Column stride of gc
        beta,                        // Beta (scaling factor for gb)
        gb,                          // Output gradient for B
        b_dim[1] as isize,           // Leading dimension of gb
        1,                           // Column stride of gb
    );
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn matmul_test() {
        let a_dim = [2, 3];
        let b_dim = [3, 4];
    
        // Example matrices A and B (flatten for raw pointers)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0];
    
        // Allocate memory for the result matrix C
        let mut c_data: Vec<f32> = vec![0.0; a_dim[0] * b_dim[1]];
    
        // Call the matrix_multiply function
        unsafe {
            matmul(
                a_data.as_ptr(),
                b_data.as_ptr(),
                c_data.as_mut_ptr(),
                a_dim,
                b_dim,
                0.0,
            );
        }
    
        // Print the result matrix C
        for i in 0..a_dim[0] {
            for j in 0..b_dim[1] {
                let index = i * b_dim[1] + j;
                eprint!("{:.2} ", c_data[index]);
            }
            eprintln!();
        }

        //panic!("");
    }

    #[test]
    fn matmul_wrt_a_test() {
        let a_dim = [2, 3];
        let b_dim = [3, 4];
    
        // Example matrices B and gradient for C (flatten for raw pointers)
        let b_data: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0];
        let gc_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    
        // Allocate memory for the gradient of A
        let mut ga_data: Vec<f32> = vec![0.0; a_dim[0] * a_dim[1]];
    
        // Call the matmul_wrt_a_internal function
        unsafe {
            matmul_wrt_a(
                gc_data.as_ptr(),
                b_data.as_ptr(),
                ga_data.as_mut_ptr(),
                a_dim,
                b_dim,
                0.0,
            )
        }
    
        // Print the gradient of A
        for i in 0..a_dim[0] {
            for j in 0..a_dim[1] {
                let index = i * a_dim[1] + j;
                print!("{:.2} ", ga_data[index]);
            }
            println!();
        }

        //panic!("");
    }

    #[test]
    fn matmul_wrt_b_test() {
        let a_dim = [2, 3];
        let b_dim = [3, 4];
    
        // Example matrices A and gradient for C (flatten for raw pointers)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gc_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    
        // Allocate memory for the gradient of B
        let mut gb_data: Vec<f32> = vec![0.0; b_dim[0] * b_dim[1]];
    
        // Call the matmul_wrt_b_internal function
        unsafe {
            matmul_wrt_b(
                a_data.as_ptr(),
                gc_data.as_ptr(),
                gb_data.as_mut_ptr(),
                a_dim,
                b_dim,
                1.0,
            );
        }
    
        // Print the gradient of B
        for i in 0..b_dim[0] {
            for j in 0..b_dim[1] {
                let index = i * b_dim[1] + j;
                print!("{:.2} ", gb_data[index]);
            }
            println!();
        }

        //panic!("");
    }
}