
use matrixmultiply::sgemm;

use crate::error::BMLSError;
use crate::error;

/// # Matrix Multiplication Operator
/// - A: Input
/// - B: Input
/// - C: Output
/// - A_Dim: dimensions of A
/// - B_Dim: dimensions of B
#[inline]
pub fn matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    a_dim: [usize; 2],
    b_dim: [usize; 2],
) -> Result<(), BMLSError> {
    let alen = a_dim[0] * a_dim[1];
    if a.len() != alen {
        return error::length_mismatch("A", a.len(), "ADim", alen);
    }

    let blen = b_dim[0] * b_dim[1];
    if b.len() != blen {
        return error::length_mismatch("B", b.len(), "BDim", blen);
    }

    let clen = a_dim[0] * b_dim[1];
    if c.len() != clen {
        return error::length_mismatch("C", c.len(), "Adim[0]*Bdim[1]", clen)
    }

    if a_dim[1] != b_dim[0] {
        return error::axis_mismatch(1, "A", a.len(), 0, "B", b.len())
    }

    unsafe {
        sgemm(
            a_dim[0],                 // M: Rows of A
            a_dim[1],                 // K: Columns of A
            b_dim[1],                 // N: Columns of B
            1.0,                      // Alpha (scaling factor for A*B)
            a.as_ptr(),
            a_dim[1] as isize,        // Leading dimension of A
            1,                        // Column stride of A
            b.as_ptr(),
            b_dim[1] as isize,        // Leading dimension of B
            1,                        // Column stride of B
            0.0,                     // Beta (scaling factor for C)
            c.as_mut_ptr(),
            b_dim[1] as isize,        // Leading dimension of C
            1,                        // Column stride of C
        );
    }

    Ok(())
}

/// # Matmul Wrt A
/// - GC: Gradient w.r.t. output C
/// - B: B in the forward op.
/// - GA: Gradient w.r.t. Input A
/// - A_Dim: Dimensions of A in the forward op
/// - B_Dim: Dimensions of B in the forward op.
#[inline]
pub fn matmul_wrt_a(
    gc: &[f32],
    b: &[f32],
    ga: &mut [f32],
    a_dim: [usize; 2],
    b_dim: [usize; 2],
) -> Result<(), BMLSError> {
    let gc_dim = [a_dim[0], b_dim[1]];
    let bt_dim = [b_dim[1], b_dim[0]];
    let ga_dim = [a_dim[0], a_dim[1]];

    let alen = a_dim[0] * a_dim[1];
    if ga.len() != alen {
        return error::length_mismatch("GA", ga.len(), "ADim", alen);
    }

    let blen = b_dim[0] * b_dim[1];
    if b.len() != blen {
        return error::length_mismatch("B", b.len(), "BDim", blen);
    }

    let clen = a_dim[0] * b_dim[1];
    if gc.len() != clen {
        return error::length_mismatch("GC", gc.len(), "Adim[0]*Bdim[1]", clen)
    }

    if gc_dim[1] != bt_dim[0] {
        return error::axis_mismatch(1, "GC", gc_dim[1], 0, "BT", bt_dim[0])
    }

    unsafe {
        sgemm(
            gc_dim[0],                    
            gc_dim[1],                    
            bt_dim[1],                    
            1.0,                         
            gc.as_ptr(),                          
            gc_dim[1] as isize,           
            1,                           
            b.as_ptr(),                             
            1,  
            bt_dim[0] as isize,                       
            1.0,                         
            ga.as_mut_ptr(),                          
            ga_dim[1] as isize,           
            1,                           
        );
    }

    Ok(())
}

/// # Matmul Wrt B
/// - A: A in the forward op.
/// - GC: Gradient w.r.t. output C
/// - GB: Gradient w.r.t. Input B
/// - A_Dim: Dimensions of A in the forward op
/// - B_Dim: Dimensions of B in the forward op.
#[inline]
pub fn matmul_wrt_b(
    a: &[f32],
    gc: &[f32],
    gb: &mut [f32],
    a_dim: [usize; 2],
    b_dim: [usize; 2],
) -> Result<(), BMLSError> {
    let at_dim = [a_dim[1], a_dim[0]];
    let gc_dim = [a_dim[0], b_dim[1]];
    let gb_dim = [b_dim[0], b_dim[1]];

    let alen = a_dim[0] * a_dim[1];
    if a.len() != alen {
        return error::length_mismatch("A", a.len(), "ADim", alen);
    }

    let blen = b_dim[0] * b_dim[1];
    if gb.len() != blen {
        return error::length_mismatch("GB", gb.len(), "BDim", blen);
    }

    let clen = a_dim[0] * b_dim[1];
    if gc.len() != clen {
        return error::length_mismatch("GC", gc.len(), "Adim[0]*Bdim[1]", clen)
    }

    if gc_dim[0] != at_dim[1] {
        return error::axis_mismatch(0, "GC", gc_dim[0], 1, "AT", at_dim[1])
    }

    unsafe {
        sgemm(
            at_dim[0],                    
            at_dim[1],                    
            gc_dim[1],                    
            1.0,                         
            a.as_ptr(),                                             
            1,         
            at_dim[0] as isize,  
            gc.as_ptr(),                           
            gc_dim[1] as isize,           
            1,                           
            1.0,                       
            gb.as_mut_ptr(),                          
            gb_dim[1] as isize,           
            1,                           
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;

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
            matmul(
                &a_data,
                &b_data,
                &mut c_data,
                a_dim,
                b_dim,
            ).unwrap();

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
            matmul_wrt_a(
                &gc_data,
                &b_data,
                &mut ga_data,
                a_dim,
                b_dim,
            ).unwrap();
    
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
            matmul_wrt_b(
                &a_data,
                &gc_data,
                &mut gb_data,
                a_dim,
                b_dim,
            ).unwrap();
    
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