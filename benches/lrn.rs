
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("lrn_inter_v1", |b| b.iter(|| {
        let x: Vec<f32> = black_box((0..2*3*255*255).map(|i| i as f32).collect());
        let mut y: Vec<f32> = black_box(x.clone());

        unsafe {
            lrn_v1(
                x.as_ptr(),
                y.as_mut_ptr(),
                black_box([2, 3, 255, 255]),
                black_box(1),
                black_box(1.0),
                black_box(1.0),
                black_box(0.000001),
                black_box(true),
            )
        }
    }));

    c.bench_function("lrn_intra_v1", |b| b.iter(|| {
        let x: Vec<f32> = black_box((0..2*3*255*255).map(|i| i as f32).collect());
        let mut y: Vec<f32> = black_box(x.clone());

        unsafe {
            lrn_v1(
                x.as_ptr(),
                y.as_mut_ptr(),
                black_box([2, 3, 255, 255]),
                black_box(1),
                black_box(1.0),
                black_box(1.0),
                black_box(0.000001),
                black_box(false),
            )
        }
    }));
}

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
pub unsafe fn lrn_v1(
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
                    let yi = n * cx * hx * wx + c * hx * wx + h * wx + w;
                    let mut sum = 0.0;

                    // found that this if statement caused no performance dif. 
                    if inter {
                        let citer = (isize::max(0, c as isize - n_size as isize) as usize)..=usize::min(cx, c + n_size);

                        for ic in citer {
                            let xi = n * cx * hx * wx + ic * hx * wx + h * wx + w;
                            sum += f32::powi(*x.add(xi), 2);
                        }
                    } else {
                        let hiter = (isize::max(0, h as isize - n_size as isize) as usize)..=usize::min(hx, h + n_size);
                        let witer = (isize::max(0, w as isize - n_size as isize) as usize)..=usize::min(wx, w + n_size);

                        for ih in hiter {
                            for iw in witer.clone() {
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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);