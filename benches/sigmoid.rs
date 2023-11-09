
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid");

    let x = black_box(vec![0.5; 100_000]);
    let mut y = black_box(vec![0.5; 100_000]);

    group.bench_function("sigmoid_ptr", |b| b.iter(|| {
        unsafe {
            let xptr = x.as_ptr();
            let yptr = y.as_mut_ptr();

            for i in 0..x.len() {
                *yptr.add(i) = 1. / (1. + f32::exp(-*xptr.add(i)))
            }
        }
    }));

    group.bench_function("sigmoid", |b| b.iter(|| {
        x.iter().zip(y.iter_mut()).for_each(|(x, y)| {
            *y = 1. / (1. + f32::exp(-x));
        })
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);