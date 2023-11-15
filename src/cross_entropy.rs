
// negative sum of y * ln(p)

#[inline]
pub fn cross_entropy(
    t: &[f32],
    p: &[f32],
    e: &mut [f32],
    g: &mut [f32],
    dim: [usize; 2],
) {
    let rows = dim[0];
    let cols = dim[1];

    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            let v = t[i*cols+j] * f32::ln(p[i*cols+j]);
            sum += v;

            g[i*cols+j] = (t[i*cols+j] - p[i*cols+j]) / cols as f32;
        }
        e[i] = -sum;
    }
}