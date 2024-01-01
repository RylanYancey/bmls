
mod adam;
mod add;
mod avg_pool;
mod axis_add;
mod axis_div;
mod axis_mul;
mod axis_sub;
mod col2im;
mod div;
mod dropout;
mod error;
mod im2col;
mod leaky_relu;
mod lrn;
mod matmul;
mod max_pool;
mod momentum;
mod mse;
mod mul;
mod reduce_mean;
mod reduce_sum;
mod relu;
mod rms_prop;
mod selu;
mod sgd;
mod sigmoid;
mod softmax;
mod sub;
mod tanh;

mod ptr;
pub use ptr::*;

pub use prelude::*;

#[cfg(not(feature = "ndarray"))]
pub mod prelude {
    use super::*;

    pub use tanh::{
        tanh,
        tanh_wrt_x,
    };

    pub use sub::{
        sub,
        sub_wrt_x1,
        sub_wrt_x2,
    };

    pub use softmax::{
        softmax,
        softmax_wrt_x,
    };

    pub use sigmoid::{
        sigmoid,
        sigmoid_wrt_x,
    };

    pub use sgd::sgd;

    pub use selu::{
        selu,
        selu_wrt_x,
    };

    pub use rms_prop::rms_prop;

    pub use relu::{
        relu,
        relu_wrt_x,
    };

    pub use reduce_sum::{
        reduce_sum,
        reduce_sum_wrt_x,
    };

    pub use reduce_mean::{
        reduce_mean,
        reduce_mean_wrt_x,
    };

    pub use mul::{
        mul,
        mul_wrt_x1,
        mul_wrt_x2,
    };

    pub use mse::mse;

    pub use max_pool::{
        max_pool,
        max_pool_wrt_a,
    };

    pub use matmul::{
        matmul,
        matmul_wrt_a,
        matmul_wrt_b,
    };

    pub use lrn::{
        lrn,
        lrn_wrt_x,
    };
    
    pub use leaky_relu::{
        leaky_relu,
        leaky_relu_wrt_x,
    };

    pub use im2col::{
        im2col,
        im2col_wrt_x,
    };
    
    pub use dropout::{
        dropout,
        dropout_wrt_x,
    };  

    pub use momentum::momentum;

    pub use div::{
        div,
        div_wrt_x1,
        div_wrt_x2,
    };

    pub use col2im::{
        col2im,
        col2im_wrt_x,
    };

    pub use axis_sub::{
        axis_sub,
        axis_sub_wrt_x1,
        axis_sub_wrt_x2,
    };

    pub use axis_mul::{
        axis_mul,
        axis_mul_wrt_x1,
        axis_mul_wrt_x2,
    };

    pub use axis_div::{
        axis_div,
        axis_div_wrt_x1,
        axis_div_wrt_x2,
    };

    pub use axis_add::{
        axis_add,
        axis_add_wrt_x1,
        axis_add_wrt_x2,
    };

    pub use avg_pool::{
        avg_pool,
        avg_pool_wrt_x,
    };

    pub use add::{
        add,
        add_wrt_x1,
        add_wrt_x2,
    };

    pub use adam::*;
}

#[cfg(feature = "ndarray")]
pub mod prelude {
    use super::*;
    use super::error::BMLSError;

    use ndarray::Array4;
    use ndarray::Dim;
    use ndarray::Axis;

    #[inline]
    pub fn adam(
        g: &Array4<f32>,
        v: &mut Array4<f32>,
        s: &mut Array4<f32>,
        w: &mut Array4<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
    ) -> Result<(), BMLSError> {
        let g = slice!(g);
        let v = slice_mut!(v);
        let s = slice_mut!(s);
        let w = slice_mut!(w);

        adam::adam(
            g, v, s, w, lr, beta1, beta2
        )
    }

    #[inline]
    pub fn add(
        x1: &Array4<f32>,
        x2: &Array4<f32>,
        y: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x1 = slice!(x1);
        let x2 = slice!(x2);
        let y = slice_mut!(y);

        add::add(
            x1, x2, y
        )
    }

    #[inline]
    pub fn add_wrt_x1(
        gy: &Array4<f32>,
        g1: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let gy = slice!(gy);
        let g1 = slice_mut!(g1);

        add::add_wrt_x1(
            gy, g1
        )
    }

    #[inline]
    pub fn add_wrt_x2(
        gy: &Array4<f32>,
        g2: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let gy = slice!(gy);
        let g2 = slice_mut!(g2);

        add::add_wrt_x2(
            gy, g2
        )
    }

    #[inline]
    pub fn avg_pool(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
        stride: [usize; 2],
        kernel: [usize; 2],
        padh: [usize; 2],
        padw: [usize; 2],
    ) -> Result<(), BMLSError> {
        let x_shape = to_array4(x.raw_dim());
    
        let x = slice!(x);
        let y = slice_mut!(y);

        avg_pool::avg_pool(
            x, y, x_shape, stride, kernel, padh, padw
        )
    }

    #[inline]
    pub fn avg_pool_wrt_x(
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
        stride: [usize; 2],
        kernel: [usize; 2],
        padh: [usize; 2],
        padw: [usize; 2],
    ) -> Result<(), BMLSError> {
        let x_shape = to_array4(gx.raw_dim());

        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        avg_pool::avg_pool_wrt_x(
            gy, gx, x_shape, stride, kernel, padh, padw
        )
    }

    #[inline]
    pub fn axis_add(
        x1: &Array4<f32>,
        x2: &Array4<f32>,
        y: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let x1_shape = to_array4(x1.raw_dim());

        let x1 = slice!(x1);
        let x2 = slice!(x2);
        let y = slice_mut!(y);

        axis_add::axis_add(x1, x2, y, x1_shape, axis.0)
    }

    #[inline]
    pub fn axis_add_wrt_x1(
        gy: &Array4<f32>,
        g1: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let gy = slice!(gy);
        let g1 = slice_mut!(g1);

        axis_add::axis_add_wrt_x1(gy, g1)
    }

    #[inline]
    pub fn axis_add_wrt_x2(
        gy: &Array4<f32>,
        g2: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let dim = to_array4(gy.raw_dim());
        let gy = slice!(gy);
        let g2 = slice_mut!(g2);

        axis_add::axis_add_wrt_x2(gy, g2, dim, axis.0)
    }

    #[inline]
    pub fn axis_div(
        x1: &Array4<f32>,
        x2: &Array4<f32>,
        y: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let dim = to_array4(y.raw_dim());
        let x1 = slice!(x1);
        let x2 = slice!(x2);
        let y = slice_mut!(y);

        axis_div::axis_div(x1, x2, y, dim, axis.0)
    }

    #[inline]
    pub fn axis_div_wrt_x1(
        x2: &Array4<f32>,
        gy: &Array4<f32>,
        g1: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let dim = to_array4(gy.raw_dim());
        let x2 = slice!(x2);
        let gy = slice!(gy);
        let g1 = slice_mut!(g1);

        axis_div::axis_div_wrt_x1(x2, gy, g1, dim, axis.0)
    }

    #[inline]
    pub fn axis_div_wrt_x2(
        x1: &Array4<f32>,
        x2: &Array4<f32>,
        gy: &Array4<f32>,
        g2: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let dim = to_array4(gy.raw_dim());
        let x1 = slice!(x1);
        let x2 = slice!(x2);
        let gy = slice!(gy);
        let g2 = slice_mut!(g2);

        axis_div::axis_div_wrt_x2(x1, x2, gy, g2, dim, axis.0)
    }

    #[inline]
    pub fn axis_mul(
        x1: &Array4<f32>,
        x2: &Array4<f32>,
        y: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let dim = to_array4(y.raw_dim());
        let x1 = slice!(x1);
        let x2 = slice!(x2);
        let y = slice_mut!(y);

        axis_mul::axis_mul(x1, x2, y, dim, axis.0)
    }

    #[inline]
    pub fn axis_mul_wrt_x1(
        x2: &Array4<f32>,
        gy: &Array4<f32>,
        g1: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let dim = to_array4(gy.raw_dim());
        let x2 = slice!(x2);
        let gy = slice!(gy);
        let g1 = slice_mut!(g1);

        axis_mul::axis_mul_wrt_x1(x2, gy, g1, dim, axis.0)
    }

    #[inline]
    pub fn axis_mul_wrt_x2(
        x1: &Array4<f32>,
        gy: &Array4<f32>,
        g2: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let dim = to_array4(gy.raw_dim());
        let x1 = slice!(x1);
        let gy = slice!(gy);
        let g2 = slice_mut!(g2);

        axis_mul::axis_mul_wrt_x2(x1, gy, g2, dim, axis.0)
    }

    #[inline]
    pub fn axis_sub(
        x1: &Array4<f32>,
        x2: &Array4<f32>,
        y: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let dim = to_array4(y.raw_dim());
        let x1 = slice!(x1);
        let x2 = slice!(x2);
        let y = slice_mut!(y);

        axis_sub::axis_sub(x1, x2, y, dim, axis.0)
    }

    #[inline]
    pub fn axis_sub_wrt_x1(
        gy: &Array4<f32>,
        g1: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let gy = slice!(gy);
        let g1 = slice_mut!(g1);

        axis_sub::axis_sub_wrt_x1(gy, g1)
    }

    #[inline]
    pub fn axis_sub_wrt_x2(
        gy: &Array4<f32>,
        g2: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let dim = to_array4(gy.raw_dim());
        let gy = slice!(gy);
        let g2 = slice_mut!(g2);

        axis_sub::axis_sub_wrt_x2(gy, g2, dim, axis.0)
    }
    
    #[inline]
    pub fn col2im(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x_dim = to_array2(x.raw_dim());
        let y_dim = to_array4(y.raw_dim());

        let x = slice!(x);
        let y = slice_mut!(y);

        col2im::col2im(x, y, x_dim, y_dim)
    }

    #[inline]
    pub fn col2im_wrt_x(
        gy: &Array4<f32>,
        gx: &mut Array4<f32>
    ) -> Result<(), BMLSError> {
        let x_dim = to_array2(gx.raw_dim());
        let y_dim = to_array4(gy.raw_dim());

        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        col2im::col2im_wrt_x(gy, gx, x_dim, y_dim)
    }

    #[inline]
    pub fn div(
        x1: &Array4<f32>,
        x2: &Array4<f32>,
        y: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x1 = slice!(x1);
        let x2 = slice!(x2);
        let y = slice_mut!(y);

        div::div(x1, x2, y)
    }

    #[inline]
    pub fn div_wrt_x1(
        x2: &Array4<f32>,
        gy: &Array4<f32>,
        g1: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x2 = slice!(x2);
        let gy = slice!(gy);
        let g1 = slice_mut!(g1);

        div::div_wrt_x1(x2, gy, g1)
    }

    #[inline]
    pub fn div_wrt_x2(
        x1: &Array4<f32>,
        x2: &Array4<f32>,
        gy: &Array4<f32>,
        g2: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x1 = slice!(x1);
        let x2 = slice!(x2);
        let gy = slice!(gy);
        let g2 = slice_mut!(g2);

        div::div_wrt_x2(x1, x2, gy, g2)
    }

    #[inline]
    pub fn dropout(
        x: &Array4<f32>,
        r: &mut Array4<f32>,
        y: &mut Array4<f32>,
        rate: f32,
    ) -> Result<(), BMLSError> {
        let x = slice!(x);
        let r = slice_mut!(r);
        let y = slice_mut!(y);

        dropout::dropout(x, r, y, rate)
    }

    #[inline]
    pub fn dropout_wrt_x(
        r: &Array4<f32>,
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
        rate: f32,
    ) -> Result<(), BMLSError> {
        let r = slice!(r);
        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        dropout::dropout_wrt_x(r, gy, gx, rate)
    }

    #[inline]
    pub fn im2col(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
        f_dim: Dim<[usize; 4]>,
        stride: [usize; 2],
        padh: [usize; 2],
        padw: [usize; 2],
    ) -> Result<(), BMLSError> {
        let x_dim = to_array4(x.raw_dim());
        let f_dim = to_array4(f_dim);
        let x = slice!(x);
        let y = slice_mut!(y);

        im2col::im2col(x, y, x_dim, f_dim, stride, padh, padw)
    }

    #[inline]
    pub fn im2col_wrt_x(
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
        f_dim: Dim<[usize; 4]>,
        stride: [usize; 2],
        padh: [usize; 2],
        padw: [usize; 2],
    ) -> Result<(), BMLSError> {
        let x_dim = to_array4(gx.raw_dim());
        let f_dim = to_array4(f_dim);

        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        im2col::im2col_wrt_x(gy, gx, x_dim, f_dim, stride, padh, padw)
    }
    
    #[inline]
    pub fn leaky_relu(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
        a: f32,
    ) -> Result<(), BMLSError> {
        let x = slice!(x);
        let y = slice_mut!(y);

        leaky_relu::leaky_relu(x, y, a)
    }

    #[inline]
    pub fn leaky_relu_wrt_x(
        x: &Array4<f32>,
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
        a: f32,
    ) -> Result<(), BMLSError> {
        let x = slice!(x);
        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        leaky_relu::leaky_relu_wrt_x(x, gy, gx, a)
    }

    #[inline]
    pub fn lrn(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
        n: usize,
        alpha: f32,
        beta: f32,
        k: f32,
        inter: bool,
    ) -> Result<(), BMLSError> {
        let x_dim = to_array4(x.raw_dim());
        let x = slice!(x);
        let y = slice_mut!(y);

        lrn::lrn(x, y, x_dim, n, alpha, beta, k, inter)
    }

    #[inline]
    pub fn lrn_wrt_x(

    ) -> Result<(), BMLSError> {
        lrn::lrn_wrt_x()
    }

    #[inline]
    pub fn matmul(
        a: &Array4<f32>,
        b: &Array4<f32>,
        c: &mut Array4<f32>, 
    ) -> Result<(), BMLSError> {
        let a_shape = to_array2(a.raw_dim());
        let b_shape = to_array2(b.raw_dim());

        let a = slice!(a);
        let b = slice!(b);
        let c = slice_mut!(c);

        matmul::matmul(
            a,
            b,
            c,
            a_shape,
            b_shape,
        )
    }

    #[inline]
    pub fn matmul_wrt_a(
        gc: &Array4<f32>,
        b: &Array4<f32>,
        ga: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let a_shape = to_array2(ga.raw_dim());
        let b_shape = to_array2(b.raw_dim());

        let gc = slice!(gc);
        let b = slice!(b);
        let ga = slice_mut!(ga);

        matmul::matmul_wrt_a(
            gc, b, ga, a_shape, b_shape,
        )
    }

    #[inline]
    pub fn matmul_wrt_b(
        a: &Array4<f32>,
        gc: &Array4<f32>,
        gb: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let a_shape = to_array2(a.raw_dim());
        let b_shape = to_array2(gb.raw_dim());

        let a = slice!(a);
        let gc = slice!(gc);
        let gb = slice_mut!(gb);

        matmul::matmul_wrt_b(
            a, gc, gb, a_shape, b_shape,
        )
    }

    #[inline]
    pub fn max_pool(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
        i: &mut Array4<usize>,
        kernel: [usize; 2],
        stride: [usize; 2],
        padh: [usize; 2],
        padw: [usize; 2],
    ) -> Result<(), BMLSError> {
        let x_dim = to_array4(x.raw_dim());
        let x = slice!(x);
        let y = slice_mut!(y);
        let i = slice_mut!(i);

        max_pool::max_pool(x, y, i, x_dim, kernel, stride, padh, padw)
    }

    #[inline]
    pub fn max_pool_wrt_x(
        i: &Array4<usize>,
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let i = slice!(i);
        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        max_pool::max_pool_wrt_a(i, gy, gx)
    }

    #[inline]
    pub fn momentum(
        g: &Array4<f32>,
        v: &mut Array4<f32>,
        w: &mut Array4<f32>,
        lr: f32,
        beta: f32,
    ) -> Result<(), BMLSError> {
        let g = slice!(g);
        let v = slice_mut!(v);
        let w = slice_mut!(w);

        momentum::momentum(g, v, w, lr, beta)
    }

    #[inline]
    pub fn mse(
        t: &Array4<f32>,
        p: &Array4<f32>,
        e: &mut Array4<f32>,
        g: &mut Array4<f32>, 
    ) -> Result<(), BMLSError> {
        let dim = to_array2(t.raw_dim());
        let t = slice!(t);
        let p = slice!(p);
        let e = slice_mut!(e);
        let g = slice_mut!(g);

        mse::mse(t, p, e, g, dim)
    }

    #[inline]
    pub fn mul(
        x1: &Array4<f32>,
        x2: &Array4<f32>,
        y: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x1 = slice!(x1);
        let x2 = slice!(x2);
        let y = slice_mut!(y);

        mul::mul(x1, x2, y)
    }

    #[inline]
    pub fn mul_wrt_x1(
        x2: &Array4<f32>,
        gy: &Array4<f32>,
        g1: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x2 = slice!(x2);
        let gy = slice!(gy);
        let g1 = slice_mut!(g1);

        mul::mul_wrt_x1(x2, gy, g1)
    }

    #[inline]
    pub fn mul_wrt_x2(
        x1: &Array4<f32>,
        gy: &Array4<f32>,
        g2: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x1 = slice!(x1);
        let gy = slice!(gy);
        let g2 = slice_mut!(g2);

        mul::mul_wrt_x2(x1, gy, g2)
    }

    #[inline]
    pub fn reduce_mean(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let x_shape = to_array4(x.raw_dim());
        let x = slice!(x);
        let y = slice_mut!(y);

        reduce_mean::reduce_mean(x, y, x_shape, axis.0)
    }

    #[inline]
    pub fn reduce_mean_wrt_x(
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let x_shape = to_array4(gx.raw_dim());
        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        reduce_mean::reduce_mean_wrt_x(gy, gx, x_shape, axis.0)
    }

    #[inline]
    pub fn reduce_sum(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let x_shape = to_array4(x.raw_dim());
        let x = slice!(x);
        let y = slice_mut!(y);

        reduce_sum::reduce_sum(x, y, x_shape, axis.0)
    }

    #[inline]
    pub fn reduce_sum_wrt_x(
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
        axis: Axis,
    ) -> Result<(), BMLSError> {
        let x_shape = to_array4(gx.raw_dim());
        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        reduce_sum::reduce_sum_wrt_x(gy, gx, x_shape, axis.0)
    }

    #[inline]
    pub fn relu(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x = slice!(x);
        let y = slice_mut!(y);

        relu::relu(x, y)
    }

    #[inline]
    pub fn relu_wrt_x(
        x: &Array4<f32>,
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x = slice!(x);
        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        relu::relu_wrt_x(x, gy, gx)
    }

    #[inline]
    pub fn rms_prop(
        g: &Array4<f32>,
        s: &mut Array4<f32>,
        w: &mut Array4<f32>,
        lr: f32,
        beta: f32,
    ) -> Result<(), BMLSError> {
        let g = slice!(g);
        let s = slice_mut!(s);
        let w = slice_mut!(w);

        rms_prop::rms_prop(g, s, w, lr, beta)
    }

    #[inline]
    pub fn selu(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
        alpha: f32,
        lambda: f32,
    ) -> Result<(), BMLSError> {
        let x = slice!(x);
        let y = slice_mut!(y);

        selu::selu(x, y, alpha, lambda)
    }

    #[inline]
    pub fn selu_wrt_x(
        x: &Array4<f32>,
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
        alpha: f32,
        lambda: f32,
    ) -> Result<(), BMLSError> {
        let x = slice!(x);
        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        selu::selu_wrt_x(x, gy, gx, alpha, lambda)
    }

    #[inline]
    pub fn sgd(
        g: &Array4<f32>,
        w: &mut Array4<f32>,
        lr: f32,
    ) -> Result<(), BMLSError> {
        let g = slice!(g);
        let w = slice_mut!(w);

        sgd::sgd(g, w, lr)
    }

    #[inline]
    pub fn sigmoid(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x = slice!(x);
        let y = slice_mut!(y);

        sigmoid::sigmoid(x, y)
    }

    #[inline]
    pub fn sigmoid_wrt_x(
        y: &Array4<f32>,
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let y = slice!(y);
        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        sigmoid::sigmoid_wrt_x(y, gy, gx)
    }

    #[inline]
    pub fn softmax(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let dim = to_array2(x.raw_dim());
        let x = slice!(x);
        let y = slice_mut!(y);

        softmax::softmax(x, y, dim)
    }

    #[inline]
    pub fn softmax_wrt_x(
        y: &Array4<f32>,
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let dim = to_array2(y.raw_dim());
        let y = slice!(y);
        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        softmax::softmax_wrt_x(y, gy, gx, dim)
    }

    #[inline]
    pub fn sub(
        x1: &Array4<f32>,
        x2: &Array4<f32>,
        y: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x1 = slice!(x1);
        let x2 = slice!(x2);
        let y = slice_mut!(y);

        sub::sub(x1, x2, y)
    }

    #[inline]
    pub fn sub_wrt_x1(
        gy: &Array4<f32>,
        g1: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let gy = slice!(gy);
        let g1 = slice_mut!(g1);

        sub::sub_wrt_x1(gy, g1)
    }

    #[inline]
    pub fn sub_wrt_x2(
        gy: &Array4<f32>,
        g2: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let gy = slice!(gy);
        let g2 = slice_mut!(g2);

        sub::sub_wrt_x2(gy, g2)
    }

    #[inline]
    pub fn tanh(
        x: &Array4<f32>,
        y: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let x = slice!(x);
        let y = slice_mut!(y);

        tanh::tanh(x, y)
    }

    pub fn tanh_wrt_x(
        y: &Array4<f32>,
        gy: &Array4<f32>,
        gx: &mut Array4<f32>,
    ) -> Result<(), BMLSError> {
        let y = slice!(y);
        let gy = slice!(gy);
        let gx = slice_mut!(gx);

        tanh::tanh_wrt_x(y, gy, gx)
    }

    #[inline]
    fn to_array2(shape: Dim<[usize; 4]>) -> [usize; 2] {
        [shape[0], shape[1]]
    }

    #[inline]
    fn to_array4(shape: Dim<[usize; 4]>) -> [usize; 4] {
        [shape[0], shape[1], shape[2], shape[3]]
    }

    use std::stringify;

    #[macro_export]
    macro_rules! slice {
        ($x:ident) => {
            $x.as_slice().ok_or(BMLSError::NdarraySliceError(String::from(stringify!($ident))))?
        }
    }

    #[macro_export]
    macro_rules! slice_mut {
        ($x:ident) => {
            $x.as_slice_mut().ok_or(BMLSError::NdarraySliceError(String::from(stringify!($ident))))?
        }
    }
}