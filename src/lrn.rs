
use rayon::prelude::*;
use crate::error::BMLSError;
use crate::error;
use crate::Ptr;

/// # Local Response Normalization
/// - X: Input
/// - Y: Output
/// - X_dim: Dimensions of X.
/// - Alpha: learning rate
/// - Beta hyperparameter
/// - N_Size: Normalization Size
/// - Inter: Specify if norm should be Inter (true) or intra (false)
/// 
/// If N_Size is 1, normalization will include
/// every neuron 1 away from the neuron, for
/// a total of a 3x3 area around the neuron to be noramlized (for intra).
/// Inter LRN would be 1x3. 
/// 
/// X and Y should have the same shape. 
#[inline]
pub fn lrn(
    x: &[f32],
    y: &mut [f32],
    x_dim: [usize; 4],
    n_size: usize,
    alpha: f32,
    beta: f32,
    k: f32,
    inter: bool,
) -> Result<(), BMLSError> {
    let (nx, cx, hx, wx) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);

    let len = nx*cx*hx*wx;
    if x.len() != len {
        return error::length_mismatch("X", x.len(), "X_Dim", len);
    }

    if y.len() != x.len() {
        return error::length_mismatch("Y", y.len(), "X", x.len());
    }

    if n_size == 0 {
        return error::invalid_lrn_size(n_size);
    }

    let x = Ptr::new(x);
    let y = Ptr::new(y);

    (0..nx).into_par_iter().for_each(|n| {
        for c in 0..cx {
            if inter {
                let citer = (isize::max(0, c as isize - n_size as isize) as usize)..=usize::min(cx, c + n_size);
                let yi = n * cx * hx * wx + c * hx * wx;
                let xi = n * cx * hx * wx;
                for h in 0..hx {
                    for w in 0..wx {
                        let yi = yi + h * wx + w;
                        let mut sum = 0.0;

                        for ic in citer.clone() {
                            let xi = xi + ic * hx * wx + h * wx + w;
                            let v = x.get_mut()[xi];
                            sum += v * v;
                        }

                        y.get_mut()[yi] = x.get_mut()[yi] / (k + (alpha * f32::powf(sum, beta)));
                    }
                }
            } else {
                for h in 0..hx {
                    let hiter = (isize::max(0, h as isize - n_size as isize) as usize)..=usize::min(hx, h + n_size);
                    let yi = n * cx * hx * wx + c * hx * wx;
                    let xi = yi;
                    for w in 0..wx {
                        let yi = yi + h * wx + w;
                        let mut sum = 0.0;

                        let witer = (isize::max(0, w as isize - n_size as isize) as usize)..=usize::min(wx, w + n_size);

                        for ih in hiter.clone() {
                            for iw in witer.clone() {
                                let xi = xi + ih * wx + iw;
                                let v = x.get_mut()[xi];
                                sum += v * v;
                            }
                        }

                        y.get_mut()[yi] = x.get_mut()[yi] / (k + (alpha * f32::powf(sum, beta)));
                    }
                }
            }
        }
    });

    Ok(())
}

#[inline]
pub fn lrn_wrt_x(

) -> Result<(), BMLSError> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lrn_inter() {
        let x = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9.,
            1., 2., 1., 2., 3., 2., 3., 4., 3.,
            2., 1., 2., 3., 2., 3., 4., 3., 4.,
            4., 2., 1., 5., 2., 1., 2., 2., 4.,
        ];

        let mut y = x.clone();

            lrn(
                &x,
                &mut y,
                [1, 4, 3, 3],
                1,
                1.0,
                1.0,
                0.0,
                true,
            ).unwrap();

        for c in 0..4 {
            println!("\n");
            for h in 0..3 {
                println!("");
                for w in 0..3 {
                    print!("{} ", y[c * 3 * 3 + h * 3 + w])
                }
            }
        }

        //panic!("")
    }

    #[test]
    fn test_lrn_intra() {
        let x = vec![
            1., 3., 2., 4., 2.,
            2., 4., 3., 5., 1.,
            2., 3., 2., 1., 3.,
            1., 3., 5., 3., 2.,
            2., 3., 4., 5., 2.,  
        ];

        let mut y = x.clone();

            lrn(
                &x, &mut y,
                [1, 1, 5, 5],
                1, 
                1.0,
                1.0,
                0.0,
                false,
            ).unwrap();

        for h in 0..5 {
            println!("");
            for w in 0..5 {
                print!("{} ", y[h * 5 + w])
            }
        }

        //panic!("")
    }
}