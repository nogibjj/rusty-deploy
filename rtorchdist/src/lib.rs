use tch::Kind;
use tch::Tensor;

pub fn get_prediction_class(output: &Tensor) -> Result<usize, actix_web::Error> {
    if output.dim() > 0 {
        let max_index = output.argmax(-1, false).to_kind(Kind::Int64);
        Ok(max_index.int64_value(&[0]) as usize)
    } else {
        Err(actix_web::Error::from(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Output tensor has 0 dimensions.",
        )))
    }
}
