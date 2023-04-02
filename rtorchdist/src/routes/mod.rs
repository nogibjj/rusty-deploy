pub mod check_image_prediction;
pub mod check_image_upload;
pub mod check_pytorch_cpu;
pub mod index;

pub use check_image_prediction::check_image_prediction as other_check_image_prediction;
pub use check_image_upload::check_image_upload as other_check_image_upload;
pub use check_pytorch_cpu::check_pytorch_cpu as other_check_pytorch_cpu;
pub use index::index as other_index;
