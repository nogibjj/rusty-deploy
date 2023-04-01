use tch::Kind;
use tch::{Device, Tensor};

/*
Model handling code
 */
pub fn get_prediction_class(output: &Tensor) -> Result<usize, actix_web::Error> {
    // Check if the output tensor has the expected dimensions.
    if output.dim() != 2 || output.size()[0] != 1 {
        return Err(actix_web::Error::from(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Output tensor has an unexpected shape.",
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

/*
Self check code to verify PyTorch works without models
 */

/*A PyTorch self-tests function */
pub fn tensor_device_cpu() -> String {
    let t = Tensor::of_slice(&[3, 1, 4]);
    let t = t.to_device(Device::Cpu);
    //log descriptive information about the tensor via log::info!
    log::info!("Tensor t: {:?}", t);
    log::info!("Tensor t size: {:?}", t.size());
    log::info!("Tensor t numel: {:?}", t.numel());
    log::info!("Tensor t dim: {:?}", t.dim());
    log::info!("Tensor t kind: {:?}", t.kind());
    log::info!("Tensor t device: {:?}", t.device());
    //return the tensor size as a string
    let size = t.size();
    let size_str = size
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    size_str
}
