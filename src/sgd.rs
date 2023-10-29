
#[inline]
pub unsafe fn sgd(
    g: *const f32,
    w: *mut f32,
    lr: f32,
    len: usize,
) {
    for i in 0..len {
        *w.add(i) -= *g.add(i) * lr
    }
}