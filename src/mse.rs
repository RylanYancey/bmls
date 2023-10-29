
/// ## Inputs
/// - T: Truth Label (N x C)
/// - P: Prediction (N x C)
/// - E: Error (N x 1) vector
/// - G: Gradient (N x C)
/// - R: Regularization Penalty
#[inline]
pub unsafe fn mse(
    t: *const f32,
    p: *const f32,
    e: *mut f32,
    g: *mut f32,
    r: f32,
    dim: [usize; 2],
) {
    let rows = dim[0];
    let cols = dim[1];
    
    for i in 0..rows {
        let t_row = t.add(i * cols);
        let p_row = p.add(i * cols);
        let g_row = g.add(i * cols);
        
        let mut sum = 0.0;

        for j in 0..cols {
            let target_val = *t_row.add(j);
            let predicted_val = *p_row.add(j);
            
            // Calculate the error (e) and gradient (g) for each element
            let error = r + (target_val - predicted_val);
            let gradient = 2.0 * error;
            
            *g_row.add(j) = gradient;

            sum += f32::powi(error, 2);
        }

        *e.add(i) = sum / dim[0] as f32;
    }
}
