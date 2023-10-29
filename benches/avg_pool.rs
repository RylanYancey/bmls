
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

use bmls::{Ptr, PtrMut};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("avg_pool");

    let mut x = black_box((0..5*3*255*255).map(|i| i as f32).collect::<Vec<f32>>()); 

    let yh = ((255 - 3 + (1 + 1)) / 1) + 1;
    let yw = ((255 - 3 + (1 + 1)) / 1) + 1;

    let mut y = black_box((0..5 * 3 * yh * yw).map(|i| i as f32).collect::<Vec<f32>>());

    group.bench_function("avg_pool_v1", |b| b.iter(|| {
        unsafe {
            avg_pool_v1(
                x.as_ptr(),
                y.as_mut_ptr(),
                black_box([5, 3, 255, 255]),
                black_box([1, 1]), black_box([3, 3]),
                black_box([1, 1]), black_box([1, 1]),
            );
        }
    }));

    group.bench_function("avg_pool_wrt_x_v1", |b| b.iter(|| {
        unsafe {
            avg_pool_wrt_x_v1(
                y.as_ptr(),
                x.as_mut_ptr(),
                black_box([5, 3, 255, 255]),
                black_box([1, 1]), black_box([3, 3]),
                black_box([1, 1]), black_box([1, 1]),
            );
        }
    }));

    group.bench_function("avg_pool_v2", |b| b.iter(|| {
        unsafe {
            avg_pool_v2(
                x.as_ptr(),
                y.as_mut_ptr(),
                black_box([5, 3, 255, 255]),
                black_box([1, 1]), black_box([3, 3]),
                black_box([1, 1]), black_box([1, 1]),
            );
        }
    }));

    group.bench_function("avg_pool_wrt_x_v2", |b| b.iter(|| {
        unsafe {
            avg_pool_wrt_x_v2(
                y.as_ptr(),
                x.as_mut_ptr(),
                black_box([5, 3, 255, 255]),
                black_box([1, 1]), black_box([3, 3]),
                black_box([1, 1]), black_box([1, 1]),
            );
        }
    }));

    group.finish()
}

/// # Avg Pooling Operation
/// - A: Input
/// - B: Output
/// - Dim: Dimensions of A. 
/// - Strides: Distance between patches
/// - Kernel: Size of the Kernel
/// - Dilations: Amount to dilate. Cannot be Zero. 
/// 
/// B Should have the height: ((input_rows - kernel_rows + (padh0 + padh1)) / stride_rows) + 1
/// B should have the width: ((input_cols - kernel_cols + (padw0 + padw1)) / stride_cols) + 1
#[inline]
pub unsafe fn avg_pool_v1(
    x: *const f32,
    y: *mut f32,
    x_dim: [usize; 4],
    stride: [usize; 2],
    kernel: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) {
    let (strideh, stridew) = (stride[0], stride[1]);
    let (kernelh, kernelw) = (kernel[0], kernel[1]);
    let (xn, xc, xh, xw) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);

    let hstart = ((x_dim[2] - kernelh + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((x_dim[3] - kernelw + (padw[0] + padw[1])) / stridew) + 1;

    let (_, yc, yh, yw) = (x_dim[0], x_dim[1], hstart, wstart);

    let k_len = kernelh * kernelw;

    for n in 0..xn {
        for c in 0..xc {
            for h in 0..hstart {
                for w in 0..wstart {

                    let mut sum = 0.0;

                    for kh in 0..kernelh {
                        for kw in 0..kernelw {

                            let xrow = ((h * strideh) + kh) as isize - padh[0] as isize;
                            let xcol = ((w * stridew) + kw) as isize - padw[0] as isize;

                            if xrow >= xh as isize || xrow < 0 || xcol >= xw as isize || xcol < 0 {
                                continue;
                            }

                            let xi = n * xc * xh * xw + c * xh * xw + xrow as usize * xw + xcol as usize; 
                            sum += *x.add(xi);
                        }
                    }
                    let yi = n * yc * yh * yw + c * yh * yw + h * yw + w;
                    *y.add(yi) = sum / k_len as f32;
                }
            }
        }
    }
}

/// Avg Pooling w.r.t. X
/// - GB: Output Gradient
/// - GA: Input Gradient
/// - Dim: Dimensions of A
/// - Strides: Distance between patches
/// - Kernel: Size of the Kernel
/// - Dilations: Amount to dilate. Cannot be Zero.
/// 
/// GB Should have the height: ((input_rows - kernel_rows + (padh0 + padh1)) / stride_rows) + 1
/// GB should have the width: ((input_cols - kernel_cols + (padw0 + padw1)) / stride_cols) + 1
/// 
/// GA is expected to be zeroed. 
#[inline]
pub unsafe fn avg_pool_wrt_x_v1(
    gy: *const f32,
    gx: *mut f32,
    x_dim: [usize; 4],
    stride: [usize; 2],
    kernel: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) {
    let (strideh, stridew) = (stride[0], stride[1]);
    let (kernelh, kernelw) = (kernel[0], kernel[1]);
    let (xn, xc, xh, xw) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);

    let hstart = ((x_dim[2] - kernelh + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((x_dim[3] - kernelw + (padw[0] + padw[1])) / stridew) + 1;

    let (_, yc, yh, yw) = (x_dim[0], x_dim[1], hstart, wstart);

    let k_len = kernelh * kernelw;

    for n in 0..xn {
        for c in 0..xc {
            for h in 0..hstart {
                for w in 0..wstart {

                    let yi = n * yc * yh * yw + c * yh * yw + h * yw + w;

                    for kh in 0..kernelh {
                        for kw in 0..kernelw {

                            let xrow = ((h * strideh) + kh) as isize - padh[0] as isize;
                            let xcol = ((w * stridew) + kw) as isize - padw[0] as isize;

                            if xrow >= xh as isize || xrow < 0 || xcol >= xw as isize || xcol < 0 {
                                continue;
                            }

                            let xi = n * xc * xh * xw + c * xh * xw + xrow as usize * xw + xcol as usize; 
                            *gx.add(xi) += *gy.add(yi) / k_len as f32;
                        }
                    }
                }
            }
        }
    }
}

/// # Avg Pooling Operation
/// - A: Input
/// - B: Output
/// - Dim: Dimensions of A. 
/// - Strides: Distance between patches
/// - Kernel: Size of the Kernel
/// - Dilations: Amount to dilate. Cannot be Zero. 
/// 
/// B Should have the height: ((input_rows - kernel_rows + (padh0 + padh1)) / stride_rows) + 1
/// B should have the width: ((input_cols - kernel_cols + (padw0 + padw1)) / stride_cols) + 1
#[inline]
pub unsafe fn avg_pool_v2(
    x: *const f32,
    y: *mut f32,
    x_dim: [usize; 4],
    stride: [usize; 2],
    kernel: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) {
    let (strideh, stridew) = (stride[0], stride[1]);
    let (kernelh, kernelw) = (kernel[0], kernel[1]);
    let (xn, xc, xh, xw) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);

    let hstart = ((x_dim[2] - kernelh + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((x_dim[3] - kernelw + (padw[0] + padw[1])) / stridew) + 1;

    let (_, yc, yh, yw) = (x_dim[0], x_dim[1], hstart, wstart);

    let k_len = kernelh * kernelw;

    let x = Ptr::new(x);
    let y = PtrMut::new(y);

    (0..xn).into_par_iter().for_each(|n| {
        for c in 0..xc {
            for h in 0..hstart {
                let xrow = h * strideh;
                let xi = n * xc * xh * xw + c * xh * xw;
                for w in 0..wstart {
                    let yi = n * yc * yh * yw + c * yh * yw + h * yw + w;
                    let xcol = w * stridew;
                    let mut sum = 0.0;
                    for kh in 0..kernelh {
                        let xrow = (xrow + kh) as isize - padh[0] as isize;
                        for kw in 0..kernelw {
                            let xcol = (xcol + kw) as isize - padw[0] as isize;
                            if xrow >= xh as isize || xrow < 0 || xcol >= xw as isize || xcol < 0 {
                                continue;
                            }
                            let xi = xi + xrow as usize * xw + xcol as usize; 
                            sum += *x.add(xi);
                        }
                    }
                    *y.add(yi) = sum / k_len as f32;
                }
            }
        }
    })
}

/// Avg Pooling w.r.t. X
/// - GB: Output Gradient
/// - GA: Input Gradient
/// - Dim: Dimensions of A
/// - Strides: Distance between patches
/// - Kernel: Size of the Kernel
/// - Dilations: Amount to dilate. Cannot be Zero.
/// 
/// GB Should have the height: ((input_rows - kernel_rows + (padh0 + padh1)) / stride_rows) + 1
/// GB should have the width: ((input_cols - kernel_cols + (padw0 + padw1)) / stride_cols) + 1
/// 
/// GA is expected to be zeroed. 
#[inline]
pub unsafe fn avg_pool_wrt_x_v2(
    gy: *const f32,
    gx: *mut f32,
    x_dim: [usize; 4],
    stride: [usize; 2],
    kernel: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
) {
    let (strideh, stridew) = (stride[0], stride[1]);
    let (kernelh, kernelw) = (kernel[0], kernel[1]);
    let (xn, xc, xh, xw) = (x_dim[0], x_dim[1], x_dim[2], x_dim[3]);

    let hstart = ((x_dim[2] - kernelh + (padh[0] + padh[1])) / strideh) + 1;
    let wstart = ((x_dim[3] - kernelw + (padw[0] + padw[1])) / stridew) + 1;

    let (_, yc, yh, yw) = (x_dim[0], x_dim[1], hstart, wstart);

    let k_len = kernelh * kernelw;

    let gy = Ptr::new(gy);
    let gx = PtrMut::new(gx);

    (0..xn).into_par_iter().for_each(|n| {
        for c in 0..xc {
            for h in 0..hstart {
                let xrow = h * strideh;
                let xi = n * xc * xh * xw + c * xh * xw;
                for w in 0..wstart {
                    let yi = n * yc * yh * yw + c * yh * yw + h * yw + w;
                    let xcol = w * stridew;
                    for kh in 0..kernelh {
                        let xrow = (xrow + kh) as isize - padh[0] as isize;
                        for kw in 0..kernelw {
                            let xcol = (xcol + kw) as isize - padw[0] as isize;
                            if xrow >= xh as isize || xrow < 0 || xcol >= xw as isize || xcol < 0 {
                                continue;
                            }
                            let xi = xi + xrow as usize * xw + xcol as usize; 
                            *gx.add(xi) += *gy.add(yi) / k_len as f32;
                        }
                    }
                }
            }
        }
    })
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);