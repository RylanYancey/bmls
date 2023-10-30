
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

/// # Matmul Wrt A
/// - GC: Gradient w.r.t. output C
/// - BT: Transpose of B in the forward op.
/// - GA: Gradient w.r.t. Input A
/// - A_Dim: Dimensions of A in the forward op
/// - B_Dim: Dimensions of B in the forward op.
/// - Beta: scaling factor for GC. 
#[inline]
pub unsafe fn matmul_wrt_a(
    gc: *const f32,
    bt: *const f32,
    ga: *mut f32,
    a_dim: [usize; 2],
    b_dim: [usize; 2],
    beta: f32,
) {

    let gc_dim = [a_dim[0], b_dim[1]];
    let bt_dim = [b_dim[1], b_dim[0]];
    let ga_dim = [a_dim[0], a_dim[1]];

    sgemm(
        gc_dim[0],                    
        gc_dim[1],                    
        bt_dim[1],                    
        1.0,                         
        gc,                          
        gc_dim[1] as isize,           
        1,                           
        bt,  
        bt_dim[1] as isize,                            
        1,                        
        beta,                         
        ga,                          
        ga_dim[1] as isize,           
        1,                           
    );
}

/// # Matmul Wrt B
/// - AT: Transpose of A in the forward op.
/// - GC: Gradient w.r.t. output C
/// - GB: Gradient w.r.t. Input B
/// - A_Dim: Dimensions of A in the forward op
/// - B_Dim: Dimensions of B in the forward op.
/// - Beta: scaling factor for GC. 
#[inline]
pub unsafe fn matmul_wrt_b(
    at: *const f32,
    gc: *const f32,
    gb: *mut f32,
    a_dim: [usize; 2],
    b_dim: [usize; 2],
    beta: f32,
) {
    let at_dim = [a_dim[1], a_dim[0]];
    let gc_dim = [a_dim[0], b_dim[1]];
    let gb_dim = [b_dim[0], b_dim[1]];

    sgemm(
        at_dim[0],                    
        at_dim[1],                    
        gc_dim[1],                    
        1.0,                         
        at,        
        at_dim[1] as isize,                                       
        1,         
        gc,                           
        gc_dim[1] as isize,           
        1,                           
        beta,                        
        gb,                          
        gb_dim[1] as isize,           
        1,                           
    );
}

#[inline]
pub unsafe fn transpose(
    x: *const f32,
    y: *mut f32,
    dim: [usize; 2],
) {
    for row in 0..dim[0] {
        for col in 0..dim[1] {
            *y.add(col * dim[0] + row) = *x.add(row * dim[1] + col);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn transpose_test() {
        let dim = [2, 4];

        // 1, 2, 3, 4,
        // 5, 6, 7, 8,
        let x = vec![1., 2., 3., 4., 5., 6., 7., 8.];

        // 1, 5,
        // 2, 6,
        // 3, 7,
        // 4, 8,
        let mut y = vec![0.0; 4*2];

        unsafe {
            transpose(
                x.as_ptr(),
                y.as_mut_ptr(),
                dim,
            )
        }

        for i in 0..4 {
            println!("");
            for j in 0..2 {
                print!("{} ", y[i * 2 + j]);
            }
        }

        //panic!("")
    }

    #[test]
    fn matmul_test() {
        let a_dim = [100, 6272];
        let b_dim = [6272, 100];
    
        // Example matrices A and B (flatten for raw pointers)
        let a_data: Vec<f32> = vec![1.0; 6272*100];
        let b_data: Vec<f32> = vec![1.0; 6272*100];
    
        // Allocate memory for the result matrix C
        let mut c_data: Vec<f32> = vec![0.0; 100*100];
    
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
        let a_dim = [100, 6272];
        let b_dim = [6272, 100];
    
        // Example matrices B and gradient for C (flatten for raw pointers)
        let b_data: Vec<f32> = vec![1.0; 6272*100];
        let gc_data: Vec<f32> = vec![1.0; 100*100];
    
        // Allocate memory for the gradient of A
        let mut ga_data: Vec<f32> = vec![0.0; 100*6272];
    
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
        let a_dim = [100, 6272];
        let b_dim = [6272, 100];
    
        // Example matrices A and gradient for C (flatten for raw pointers)
        let a_data: Vec<f32> = vec![1.0; 100*6272];
        let gc_data: Vec<f32> = vec![1.0; 100*100];
    
        // Allocate memory for the gradient of B
        let mut gb_data: Vec<f32> = vec![0.0; 6272*100];
    
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