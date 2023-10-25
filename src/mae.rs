
#[inline]
pub unsafe fn mae(
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
        
        for j in 0..cols {
            let target_val = *t_row.offset(j as isize);
            let predicted_val = *p_row.offset(j as isize);
            
            // Calculate the absolute error (e) and gradient (g) for each element
            let error = target_val - predicted_val;
            let gradient = if error > 0.0 { 1.0 } else { -1.0 };
            
            *e_row.offset(j as isize) = error;
            *g_row.offset(j as isize) = gradient;
        }
    }
}