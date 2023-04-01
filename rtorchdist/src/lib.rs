use actix_web::web::BytesMut;
use futures::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use std::env;
use tch::vision::imagenet;
use tch::Kind;
use tch::{Device, IValue, Tensor};
use log::info;

//custom errors
pub mod cerror;
use cerror::CustomError;
/*
Model handling code
*/

#[derive(Serialize, Deserialize, Debug)]
pub struct Prediction {
    probabilities: Vec<f64>,
    classes: Vec<String>,
}

pub async fn convert_payload_to_vec_u8(
    mut payload: actix_multipart::Multipart,
) -> Result<std::vec::Vec<u8>, cerror::CustomError> {
    let mut bytes = BytesMut::new();
    while let Some(field) = payload.next().await {
        let field = field.map_err(cerror::CustomError::from)?;
        let data = field
            .map(|res| res.map_err(cerror::CustomError::from))
            .try_fold(BytesMut::new(), |mut acc, chunk| async move {
                acc.extend_from_slice(&chunk);
                Ok(acc)
            })
            .await?;
        bytes.extend_from_slice(&data);
    }
    Ok::<std::vec::Vec<u8>, cerror::CustomError>(bytes.to_vec())
}

pub fn preprocess_image(image_data: Vec<u8>) -> Result<Tensor, CustomError> {
    info!("Received image data with length {}", image_data.len());
    let image = image::load_from_memory(&image_data)
        .map_err(CustomError::from)?
        .to_rgb8();

    // Save the preprocessed image as a temporary file
    info!("Saving the preprocessed image as a temporary file");
    let temp_image_path = std::env::temp_dir().join("temp_image.jpg");
    image.save(&temp_image_path).map_err(CustomError::from)?;

    // Preprocess the image to match the model's requirements
    let image = tch::vision::imagenet::load_image_and_resize224(temp_image_path)?;
    info!("Preprocessed image with shape {:?}", image.size());
    Ok(image.unsqueeze(0))
}

pub fn get_model() -> Result<tch::CModule, CustomError> {
    let model_path = match env::var("MODEL_PATH") {
        Ok(path) => path,
        Err(_) => "model/resnet34.ot".to_string(),
    };
    log::info!("Loading model from path: {}", model_path);
    tch::CModule::load(model_path).map_err(CustomError::from)
}

pub fn predict_image(tensor: Tensor, model: &tch::CModule) -> Result<Prediction, CustomError> {
    // Apply the forward pass of the model to get the logits
    //log each step
    info!("Applying forward pass of the model to get the logits");
    let output = model
        .forward_is(&[IValue::from(tensor)])
        .map_err(CustomError::from)?;

    // Convert the logits to probabilities using softmax
    info!("Converting the logits to probabilities using softmax");
    let probabilities = match output {
        IValue::Tensor(tensor) => tensor.softmax(-1, Kind::Float),
        _ => return Err(CustomError::new("Output is not a Tensor!")),
    };

    // Get the index and probability of the most likely class
    info!("Getting the index and probability of the most likely class");
    let top_result = imagenet::top(&probabilities, 1); // Bind the result to a variable
    let (class, confidence) = top_result.first().unwrap();

    // Parse confidence as f64
    let confidence_f64 = confidence.parse::<f64>().map_err(CustomError::from)?;
    //log the result
    info!("Prediction result: {:?}", class);

    Ok(Prediction {
        probabilities: vec![confidence_f64],
        classes: vec![(*class as usize).to_string()],
    })
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
