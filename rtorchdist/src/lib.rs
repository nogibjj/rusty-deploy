use tch::Kind;
use tch::Tensor;

pub fn get_prediction_class(output: &Tensor) -> Result<usize, actix_web::Error> {
    // Check if the output tensor has any dimensions.
    if output.dim() <= 0 {
        return Err(actix_web::Error::from(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Output tensor has 0 dimensions.",
        )));
    }

    // Get the maximum index.
    let max_index = output.argmax(-1, false).to_kind(Kind::Int64);

    // Check if the maximum index tensor is well-formed.
    if max_index.dim() != 1 || max_index.size()[0] != 1 {
        return Err(actix_web::Error::from(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Unexpected shape for max_index tensor.",
        )));
    }

    // Get the integer value from the maximum index tensor.
    let int_value = max_index.int64_value(&[0]);

    Ok(int_value as usize)
}
