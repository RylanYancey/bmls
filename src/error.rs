
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BMLSError {
    #[error("Lengths of ({0}:{1}) and ({2}:{3}) do not match.")]
    LengthMismatch(String, usize, String, usize),
    #[error("Invalid Kernel. Dimensions cannot be Zero or larger than the image! (dim: {0}, {1}, {2}, {3})")]
    InvalidKernelDim(usize, usize, usize, usize),
    #[error("Invalid Strides. Strides cannot be Zero! (strides: {0}, {1})")]
    InvalidStrides(usize, usize),
    #[error("Axis {0} of {1} with len {2} must match axis {3} of {4} with len {5}")]
    AxisMismatch(usize, String, usize, usize, String, usize),
    #[error("The Dropout Rate must be between 0 and 1! (rate: {0}")]
    InvalidDropoutRate(f32),
    #[error("LRN N_Size must not be zero.")]
    InvalidLRNSize(usize),
    #[error("The '0'(or N) dimension of {0} must match the '1'(or C) dimension of {2}. ({0} len: {1}) ({2} len: {3})")]
    Col2ImChannelMismatch(String, usize, String, usize),
    #[cfg(feature = "ndarray")]
    #[error("Failed to convert Array4 with name {0} to slice!")]
    NdarraySliceError(String),
}

pub(crate) fn length_mismatch(a_name: &str, a_len: usize, b_name: &str, b_len: usize) -> Result<(), BMLSError> {
    Err(BMLSError::LengthMismatch(a_name.to_owned(), a_len, b_name.to_owned(), b_len))
}

pub(crate) fn invalid_kernel_dim(dim: [usize; 4]) -> Result<(), BMLSError> {
    Err(BMLSError::InvalidKernelDim(dim[0], dim[1], dim[2], dim[3]))
}

pub(crate) fn invalid_strides(h: usize, w: usize) -> Result<(), BMLSError> {
    Err(BMLSError::InvalidStrides(h, w))
}

pub(crate) fn axis_mismatch(
    a_axis: usize, a_name: &str, a_len: usize, 
    b_axis: usize, b_name: &str, b_len: usize
) -> Result<(), BMLSError> {
    Err(BMLSError::AxisMismatch(a_axis, a_name.to_owned(), a_len, b_axis, b_name.to_owned(), b_len))
}

pub(crate) fn invalid_dropout_rate(rate: f32) -> Result<(), BMLSError> {
    Err(BMLSError::InvalidDropoutRate(rate))
}

pub(crate) fn invalid_lrn_size(size: usize) -> Result<(), BMLSError> {
    Err(BMLSError::InvalidLRNSize(size))
}

pub(crate) fn col2im_channel_mismatch(a_name: &str, a_len: usize, b_name: &str, b_len: usize) -> Result<(), BMLSError> {
    Err(BMLSError::Col2ImChannelMismatch(a_name.to_owned(), a_len, b_name.to_owned(), b_len))
}