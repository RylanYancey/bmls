

use std::time::{SystemTime, UNIX_EPOCH};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("reduce_mean_v1", |b| b.iter(|| {

        let t = 
            SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() % 4;
            
        // 10 x 10 x 10 x 10
        let x: Vec<f32> = black_box((0..10*10*10*10).map(|i| i as f32).collect());
        // 10 x 10 x 10
        let mut y: Vec<f32> = black_box((0..10*10*10).map(|i| i as f32).collect());

        unsafe {
            reduce_mean_v1(
                x.as_ptr(),
                y.as_mut_ptr(),
                [10, 10, 10, 10],
                t as usize,
                0.0,
            )
        }
    }));

    c.bench_function("reduce_mean_wrt_x_v1", |b| b.iter(|| {

        let t = 
            SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() % 4;
            
        // 10 x 10 x 10 x 10
        let mut gx: Vec<f32> = black_box((0..10*10*10*10).map(|i| i as f32).collect());
        // 10 x 10 x 10
        let gy: Vec<f32> = black_box((0..10*10*10).map(|i| i as f32).collect());

        unsafe {
            reduce_mean_wrt_x_v1(
                gy.as_ptr(),
                gx.as_mut_ptr(),
                [10, 10, 10, 10],
                t as usize,
                0.0,
            )
        }
    }));
}

/// # Reduce Mean Operator
/// - X: Input
/// - Y: Output
/// - X_shape: shape of X
/// - Axis: Axis to sum
/// 
/// Y Shape is the same as X, but with the specified Axis set to 1. 
#[inline]
pub unsafe fn reduce_mean_v1(
    x: *const f32,
    y: *mut f32,
    x_shape: [usize; 4],
    axis: usize,
    beta: f32,
) {
    let xd = x_shape;
    let mut yd = x_shape;
    yd[axis] = 1;

    for i in 0..(yd[0] * yd[1] * yd[2] * yd[3]) {
        *y.add(i) *= beta;
    }

    for n in 0..yd[0] {
        for c in 0..yd[1] {
            for h in 0..yd[2] {
                for w in 0..yd[3] {
                    let yi = n * yd[1] * yd[2] * yd[3] + c * yd[2] * yd[3] + h * yd[3] + w;
                    let yptr = y.add(yi);

                    for b in 0..xd[axis] {
                        let mut i = [n, c, h, w];
                        i[axis] = b;

                        let xi = i[0] * xd[1] * xd[2] * xd[3] + i[1] * xd[2] * xd[3] + i[2] * xd[3] + i[3];
                        *yptr += *x.add(xi);
                    }

                    *yptr /= xd[axis] as f32;
                }
            }
        }
    }
}

/// # Reduce Mean W.r.t. X
/// - GY: Gradient w.r.t. Output Y
/// - GX: Gradient w.r.t. Input X
/// - X_Shape: Shape of X in the forward op.
/// - Axis: Axis to be reduced
/// - Beta: scaling factor for y. 
#[inline]
pub unsafe fn reduce_mean_wrt_x_v1(
    gy: *const f32,
    gx: *mut f32,
    x_shape: [usize; 4],
    axis: usize,
    beta: f32,
) {
    let xd = x_shape;
    let mut yd = x_shape;
    yd[axis] = 1;

    let len = xd[axis] as f32;

    for i in 0..(xd[0] * xd[1] * xd[2] * xd[3]) {
        *gx.add(i) *= beta;
    }

    for n in 0..yd[0] {
        for c in 0..yd[1] {
            for h in 0..yd[2] {
                for w in 0..yd[3] {
                    let yi = n * yd[1] * yd[2] * yd[3] + c * yd[2] * yd[3] + h * yd[3] + w;
                    let val = *gy.add(yi) / len;

                    for b in 0..xd[axis] {
                        let mut i = [n, c, h, w];
                        i[axis] = b;

                        let xi = i[0] * xd[1] * xd[2] * xd[3] + i[1] * xd[2] * xd[3] + i[2] * xd[3] + i[3];
                        *gx.add(xi) += val; 
                    }
                }
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);