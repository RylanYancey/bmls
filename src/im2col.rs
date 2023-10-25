
/// # Im2col Operation
/// - X: Input
/// - Y: Output
/// - X_dim: Dimensions of X
/// - F_dim: Dimensions of Filter F
/// - Stride: h, w, strides of the filter
/// - Padh: height padding
/// - Padw: width padding
/// 
/// Y Height: fc * fh * fw
/// 
/// Y Width: (((xh - fh + (padh.0 + padh.1)) / strideh) + 1) * (((xw - fw + (padw.0 + padw.1)) / stridew) + 1) * fn * xn
#[inline]
pub unsafe fn im2col(
    x: *const f32,
    y: *mut f32,
    x_dim: [usize; 4],
    f_dim: [usize; 4],
    stride: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) {
    let (nx, cx, hx, wx) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);
    let (nf, cf, hf, wf) = (f_dim[0], f_dim[1], f_dim[2], f_dim[3]);
    let (strideh, stridew) = (stride[0], stride[1]);
    let hstart = ((hx - hf + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((wx - wf + (padw[0] + padw[1])) / stridew) + 1;
    // size of the output Y
    let (_, cy) = (hf * wf * cf, hstart * wstart * nf * nx);

    for n in 0..nx {
        for h in 0..hstart {
            for w in 0..wstart {
                for c in 0..cx {
                    for m in 0..nf {
                        for kh in 0..hf {
                            for kw in 0..wf {

                                let xrow = ((h * strideh) + kh) as isize - padh[0] as isize;
                                let xcol = ((w * stridew) + kw) as isize - padw[0] as isize;

                                let row = (kh * wf + kw) + (wf * hf * c);
                                let col = (h * wstart + w) * (m + 1);
                                let yi = row * cy + col;

                                if xrow >= hx as isize || xrow < 0 || xcol >= wx as isize || xcol < 0 {
                                    *y.add(yi) = 0.0;
                                    continue;
                                }

                                let xi = n * cx * hx * wx + c * hx * wx + xrow as usize * wx + xcol as usize; 

                                *y.add(yi) = *x.add(xi);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// # Im2col w.r.t. X
/// - GY: Gradient w.r.t. output Y
/// - GX: Gradient w.r.t. input X
/// - X_dim: dimensions of X in the forward op
/// - F_dim: dimensions of F in the forward op
/// - stride: H and W strides of the filter.
/// - Padh: height padding
/// - Padw: width padding
/// - Beta: scaling factor
/// 
/// GY Height: fc * fh * fw
/// 
/// GY Width: (((xh - fh + (padh.0 + padh.1)) / strideh) + 1) * (((xw - fw + (padw.0 + padw.1)) / stridew) + 1) * fn * xn
#[inline]
pub unsafe fn im2col_wrt_x(
    gy: *const f32,
    gx: *mut f32,
    x_dim: [usize; 4],
    f_dim: [usize; 4],
    stride: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
    beta: f32,
) {
    let (nx, cx, hx, wx) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);
    let (nf, cf, hf, wf) = (f_dim[0], f_dim[1], f_dim[2], f_dim[3]);
    let (strideh, stridew) = (stride[0], stride[1]);
    let hstart = ((hx - hf + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((wx - wf + (padw[0] + padw[1])) / stridew) + 1;
    // size of the output Y
    let (_, cy) = (hf * wf * cf, hstart * wstart * nf * nx);

    for i in 0..(nx * cx * hx * wx) {
        *gx.add(i) *= beta;
    }

    for n in 0..nx {
        for h in 0..hstart {
            for w in 0..wstart {
                for c in 0..cx {
                    for m in 0..nf {
                        for kh in 0..hf {
                            for kw in 0..wf {

                                let xrow = ((h * strideh) + kh) as isize - padh[0] as isize;
                                let xcol = ((w * stridew) + kw) as isize - padw[0] as isize;

                                let row = (kh * wf + kw) + (wf * hf * c);
                                let col = (h * wstart + w) * (m + 1);
                                let yi = row * cy + col;

                                if xrow >= hx as isize || xrow < 0 || xcol >= wx as isize || xcol < 0 {
                                    continue;
                                }

                                let xi = n * cx * hx * wx + c * hx * wx + xrow as usize * wx + xcol as usize; 

                                *gx.add(xi) += *gy.add(yi);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn im2col() {
        // 4x4 (3 channels)
        let x: Vec<f32> = (1..=48).map(|i| i as f32).collect();
        // 12x9
        let mut y = vec![0.0; 12 * 9];

        unsafe {
            super::im2col(
                x.as_ptr(),
                y.as_mut_ptr(),
                [1, 3, 4, 4],
                [1, 3, 2, 2],
                [1, 1],
                [0, 0],
                [0, 0],
            );
        }

        for row in 0..12 {
            println!("");
            for col in 0..9 {
                print!("{}, ", y[row * 9 + col])
            }
        }
        println!("");

        //panic!("");
    }
}