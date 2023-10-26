
/// # Local Response Normalization
/// - X: Input
/// - Y: Output
/// - X_dim: Dimensions of X.
/// - Alpha: learning rate
/// - Beta hyperparameter
/// - N_Size: Normalization Size
/// 
/// If N_Size is 1, normalization will include
/// every neuron 1 away from the neuron, for
/// a total of a 3x3 area around the neuron to be noramlized (for intra).
/// Inter LRN would be 1x3. 
/// 
/// X and Y should have the same shape. 
#[inline]
pub unsafe fn lrn(
    x: *const f32,
    y: *mut f32,
    x_dim: [usize; 4],
    n_size: usize,
    alpha: f32,
    beta: f32,
    k: f32,
    inter: bool,
) {
    let (nx, cx, hx, wx) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);

    for n in 0..nx {
        for c in 0..cx {
            for h in 0..hx {
                for w in 0..wx {
                    // index of Y to assign to
                    let yi = n * cx * hx * wx + c * hx * wx + h * wx + w;
                    let mut sum = 0.0;

                    // If Inter-channel Normalization is selected, iterate the channels.
                    if inter {
                        // clamp the channel iterator to between 0, cx
                        let citer = (isize::max(0, c as isize - n_size as isize) as usize)..=usize::min(cx, c + n_size);

                        // take the sum of the squares
                        for ic in citer {
                            // index of X to add to the sum. 
                            let xi = n * cx * hx * wx + ic * hx * wx + h * wx + w;
                            sum += f32::powi(*x.add(xi), 2);
                        }
                    // if Intra-channel normalization is selected, iterate the H and W.
                    } else {
                        // clamp the iterators between 0 and the hx or wx, respectively.
                        let hiter = (isize::max(0, h as isize - n_size as isize) as usize)..=usize::min(hx, h + n_size);
                        let witer = (isize::max(0, w as isize - n_size as isize) as usize)..=usize::min(wx, w + n_size);

                        // take the sum of the squares
                        for ih in hiter {
                            for iw in witer.clone() {
                                // index of X to add to the sum. 
                                let xi = n * cx * hx * wx + c * hx * wx + ih * wx + iw;
                                sum += f32::powi(*x.add(xi), 2);
                            }
                        }
                    }

                    *y.add(yi) = *x.add(yi) / (k + (alpha * f32::powf(sum, beta)));
                }
            }
        }
    }
}

#[inline]
pub unsafe fn lrn_wrt_x(

) {
    
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

        unsafe {
            lrn(
                x.as_ptr(),
                y.as_mut_ptr(),
                [1, 4, 3, 3],
                1,
                1.0,
                1.0,
                0.0,
                true,
            )
        }

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

        unsafe {
            lrn(
                x.as_ptr(),
                y.as_mut_ptr(),
                [1, 1, 5, 5],
                1, 
                1.0,
                1.0,
                0.0,
                false,
            )
        }

        for h in 0..5 {
            println!("");
            for w in 0..5 {
                print!("{} ", y[h * 5 + w])
            }
        }

        //panic!("")
    }
}