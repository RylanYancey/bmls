
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

use bmls::{Ptr, PtrMut};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("im2col");
        group.sample_size(30);
        
    let mut x: Vec<f32> = black_box((0..4*3*255*255).map(|i| i as f32).collect());

    let yh = 3 * 3 * 3;
    let yw = (((255 - 3 + (1 + 1)) / 1) + 1) * (((255 - 3 + (1 + 1)) / 1) + 1) * 5 * 4;

    let mut y: Vec<f32> = black_box((0..(yh * yw)).map(|i| i as f32).collect());

    group.bench_function("im2col_v1", |b| b.iter(|| {
        unsafe {
            im2col_v1(
                x.as_ptr(),
                y.as_mut_ptr(),
                black_box([4, 3, 255, 255]),
                black_box([5, 3, 3, 3]),
                black_box([1, 1]),
                black_box([1, 1]),
                black_box([1, 1]),
            );
        }
    }));

    group.bench_function("im2col_wrt_x_v1", |b| b.iter(|| {
        unsafe {
            im2col_wrt_x_v1(
                y.as_ptr(),
                x.as_mut_ptr(),
                black_box([4, 3, 255, 255]),
                black_box([5, 3, 3, 3]),
                black_box([1, 1]),
                black_box([1, 1]),
                black_box([1, 1]),
                0.0
            );
        }
    }));

    group.bench_function("im2col_v2", |b| b.iter(|| {
        unsafe {
            im2col_v2(
                x.as_ptr(),
                y.as_mut_ptr(),
                black_box([4, 3, 255, 255]),
                black_box([5, 3, 3, 3]),
                black_box([1, 1]),
                black_box([1, 1]),
                black_box([1, 1]),
            );
        }
    }));

    group.bench_function("im2col_wrt_x_v2", |b| b.iter(|| {
        unsafe {
            im2col_wrt_x_v2(
                y.as_ptr(),
                x.as_mut_ptr(),
                black_box([4, 3, 255, 255]),
                black_box([5, 3, 3, 3]),
                black_box([1, 1]),
                black_box([1, 1]),
                black_box([1, 1]),
                0.0,
            );
        }
    }));

    group.bench_function("im2col_v3", |b| b.iter(|| {
        unsafe {
            im2col_v3(
                x.as_ptr(),
                y.as_mut_ptr(),
                black_box([4, 3, 255, 255]),
                black_box([5, 3, 3, 3]),
                black_box([1, 1]),
                black_box([1, 1]),
                black_box([1, 1]),
            );
        }
    }));

    group.finish()
}


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
pub unsafe fn im2col_v1(
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
pub unsafe fn im2col_wrt_x_v1(
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
pub unsafe fn im2col_v2(
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

    let x = Ptr::new(x);
    let y = PtrMut::new(y);

    (0..nx).into_par_iter().for_each(|n| {
        for h in 0..hstart {
            for w in 0..wstart {
                // 4% improvement swapping
                // the order of m and c. 
                for m in 0..nf {
                    // moved from inner loop to here
                    let col = (h * wstart + w) * (m + 1);
                    for c in 0..cx {
                        // moving index calculations out of
                        // inner loop rendered 30% improvement.
                        let xrow = h * strideh;
                        let xcol = w * stridew;
                        let row = wf * hf * c;
                        let xi =  n * cx * hx * wx + c * hx * wx;

                        // swapping the order of the kernel iteration
                        // rendererd a 16% improvement.
                        for kw in 0..wf {
                            for kh in 0..hf {

                                let xrow = (xrow + kh) as isize - padh[0] as isize;
                                let xcol = (xcol + kw) as isize - padw[0] as isize;

                                let row = (kh * wf + kw) + row;
                                let yi = row * cy + col;

                                if xrow >= hx as isize || xrow < 0 || xcol >= wx as isize || xcol < 0 {
                                    *y.add(yi) = 0.0;
                                    continue;
                                }

                                let xi = xi + xrow as usize * wx + xcol as usize; 

                                *y.add(yi) = *x.add(xi);
                            }
                        }
                    }
                }
            }
        }
    })
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
pub unsafe fn im2col_wrt_x_v2(
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

    let gy = Ptr::new(gy);
    let gx = PtrMut::new(gx);

    (0..nx).into_par_iter().for_each(|n| {
        for h in 0..hstart {
            for w in 0..wstart {
                for m in 0..nf {
                    let col = (h * wstart + w) * (m + 1);
                    for c in 0..cx {
                        let xrow = h * strideh;
                        let xcol = w * stridew;
                        let row = wf * hf * c;
                        let xi =  n * cx * hx * wx + c * hx * wx;
                        for kh in 0..hf {
                            for kw in 0..wf {

                                let xrow = (xrow + kh) as isize - padh[0] as isize;
                                let xcol = (xcol + kw) as isize - padw[0] as isize;

                                let row = (kh * wf + kw) + row;
                                let yi = row * cy + col;

                                if xrow >= hx as isize || xrow < 0 || xcol >= wx as isize || xcol < 0 {
                                    continue;
                                }

                                let xi = xi + xrow as usize * wx + xcol as usize; 

                                *gx.add(xi) += *gy.add(yi);
                            }
                        }
                    }
                }
            }
        }
    })
}

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
pub unsafe fn im2col_v3(
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

    let x = Ptr::new(x);
    let y = PtrMut::new(y);

    (0..nx).into_par_iter().for_each(|n| {
        (0..hstart).into_par_iter().for_each(|h| {
            for w in 0..wstart {
                // 4% improvement swapping
                // the order of m and c. 
                for m in 0..nf {
                    // moved from inner loop to here
                    let col = (h * wstart + w) * (m + 1);
                    for c in 0..cx {
                        // moving index calculations out of
                        // inner loop rendered 30% improvement.
                        let xrow = h * strideh;
                        let xcol = w * stridew;
                        let row = wf * hf * c;
                        let xi =  n * cx * hx * wx + c * hx * wx;

                        // swapping the order of the kernel iteration
                        // rendererd a 16% improvement.
                        for kw in 0..wf {
                            for kh in 0..hf {

                                let xrow = (xrow + kh) as isize - padh[0] as isize;
                                let xcol = (xcol + kw) as isize - padw[0] as isize;

                                let row = (kh * wf + kw) + row;
                                let yi = row * cy + col;

                                if xrow >= hx as isize || xrow < 0 || xcol >= wx as isize || xcol < 0 {
                                    *y.add(yi) = 0.0;
                                    continue;
                                }

                                let xi = xi + xrow as usize * wx + xcol as usize; 

                                *y.add(yi) = *x.add(xi);
                            }
                        }
                    }
                }
            }
        })
    })
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);