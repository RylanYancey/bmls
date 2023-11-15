
#[derive(Copy, Clone)]
pub struct Ptr<T>(*mut T, usize);

impl<T> Ptr<T> {
    pub fn new(p: &[T]) -> Self {
        Self (p.as_ptr() as *mut T, p.len())
    }

    #[inline]
    pub fn get_mut(&self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.0, self.1)
        }
    }
}

unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}