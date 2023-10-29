
use std::ops::{Deref, DerefMut};

#[derive(Copy, Clone)]
pub struct Ptr<T>(*const T);

impl<T> Ptr<T> {
    pub fn new(p: *const T) -> Self {
        Self(p)
    }

    pub fn add(self, count: usize) -> Self {
        unsafe { Self(self.0.add(count)) }
    }
}

impl<T> Deref for Ptr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

#[derive(Copy, Clone)]
pub struct PtrMut<T>(*mut T);

impl<T> PtrMut<T> {
    pub fn new(p: *mut T) -> Self {
        Self(p)
    }

    pub fn add(self, count: usize) -> Self {
        unsafe { Self(self.0.add(count)) }
    }
}

impl<T> Deref for PtrMut<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl<T> DerefMut for PtrMut<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut*self.0 }
    }
}

unsafe impl<T> Send for PtrMut<T> {}
unsafe impl<T> Sync for PtrMut<T> {}