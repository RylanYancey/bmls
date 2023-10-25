
/// ## Inputs
/// - T: Target (N x C)
/// - P: Prediction (N x C)
/// - E: Error (N x 1) vector
/// - G: Gradient (N x C)
#[inline]
pub unsafe fn mse(
    t: *const f32,
    p: *const f32,
    e: *mut f32,
    g: *mut f32,
    dim: [usize; 2],
) {
    let rows = dim[0];
    let cols = dim[1];
    
    for i in 0..rows {
        let t_row = t.offset((i * cols) as isize);
        let p_row = p.offset((i * cols) as isize);
        let e_row = e.offset((i * cols) as isize);
        let g_row = g.offset((i * cols) as isize);
        
        let mut sum = 0.0;

        for j in 0..cols {
            let target_val = *t_row.offset(j as isize);
            let predicted_val = *p_row.offset(j as isize);
            
            // Calculate the error (e) and gradient (g) for each element
            let error = target_val - predicted_val;
            let gradient = -2.0 * (predicted_val - target_val);
            
            *g_row.offset(j as isize) = gradient;

            sum += f32::powi(error, 2);
        }

        *e_row = sum / dim[0] as f32;
    }
}
