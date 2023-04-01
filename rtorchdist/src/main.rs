use actix_multipart::{Field, Multipart};
use actix_web::get;
use actix_web::http::header::ContentDisposition;
use actix_web::{middleware::Logger, post, App, HttpServer};
use actix_web::{Error, HttpResponse, Result};
use futures::StreamExt;
use serde::Serialize;
use std::env;

use tch::{Device, IValue, Kind, Tensor};

//custom errors
mod cerror;
use cerror::CustomError;

//from lib
use rtorchdist::get_prediction_class;
use rtorchdist::tensor_device_cpu;

#[derive(Serialize)]
struct Prediction {
    class: usize,
    confidence: f64,
}

#[get("/")]
async fn index() -> HttpResponse {
    HttpResponse::Ok().content_type("text/plain").body("Send an image payload using curl with the following command:\ncurl -X POST -H \"Content-Type: multipart/form-data\" -F \"image=@/path/to/your/image.jpg\" http://127.0.0.1:8080/predict")
}

fn preprocess_image(image_data: Vec<u8>) -> Result<Tensor, CustomError> {
    println!("Received image data with length {}", image_data.len());
    let image = image::load_from_memory(&image_data)
        .map_err(CustomError::from)?
        .to_rgb8();

    // Resize the image to 256x256
    let mut resized =
        image::imageops::resize(&image, 256, 256, image::imageops::FilterType::Triangle);

    // Center crop the image to 224x224
    let (width, height) = (224, 224);
    let crop_x = (resized.width() - width) / 2;
    let crop_y = (resized.height() - height) / 2;
    let cropped = image::imageops::crop(&mut resized, crop_x, crop_y, width, height);

    let cropped_raw = cropped.to_image().into_raw();
    let norm_img: Vec<f32> = cropped_raw.into_iter().map(|v| v as f32 / 255.0).collect();
    log::info!("norm_img: {:.20?}", norm_img);
    let byte_data: Vec<u8> = bytemuck::cast_slice(&norm_img).to_vec();

    let tensor = Tensor::of_data_size(&byte_data, &[1, 224, 224, 3], tch::Kind::Float)
        .permute(&[0, 3, 1, 2])
        .to_device(Device::Cpu);
    log::info!(
        "Created tensor with size {:?} in function: {}",
        tensor.size(),
        module_path!()
    );
    log::info!("Tensor:\n{} in function: {}", tensor, module_path!());
    Ok(tensor)
}

fn get_model() -> Result<tch::CModule, CustomError> {
    let model_path = match env::var("MODEL_PATH") {
        Ok(path) => path,
        Err(_) => "model/resnet34.ot".to_string(),
    };
    log::info!("Loading model from path: {}", model_path);
    tch::CModule::load(model_path).map_err(CustomError::from)
}

/*
read image data from the payload and preprocess it
 */
async fn read_image_data_and_preprocess(mut payload: Multipart) -> Result<Tensor, CustomError> {
    //create log message that describes the payload as well as the name of the function
    log::info!(
        "Reading image data from payload in function: {}",
        module_path!()
    );
    let mut field_opt: Option<Field> = None;

    while let Some(field) = payload.next().await {
        let field = field.map_err(CustomError::from)?;
        let cd = field.content_disposition();

        if let Some(name) = get_name_from_content_disposition(cd) {
            if name == "image" {
                field_opt = Some(field);
                break;
            }
        }
    }

    let mut field = field_opt
        .ok_or_else(|| CustomError::from("Failed to find the `image` field in the payload"))?;

    let mut buf = Vec::new();
    while let Some(chunk) = field.next().await {
        let data = chunk.map_err(CustomError::from)?;
        buf.extend_from_slice(&data);
    }
    let image_data = buf;

    preprocess_image(image_data)
}

fn get_name_from_content_disposition(content_disposition: &ContentDisposition) -> Option<String> {
    content_disposition.get_name().map(ToString::to_string)
}

async fn predict_image(tensor: Tensor, model: &tch::CModule) -> Result<Prediction, CustomError> {
    // Move tensor to CPU
    let input_tensor = tensor.to_device(Device::Cpu);

    // Forward pass through model
    let output = model.forward_is(&[IValue::from(input_tensor)])?;

    let prediction_tensor = match output {
        IValue::Tensor(tensor) => tensor,
        _ => return Err(CustomError::new("Output is not a Tensor!")),
    };

    // Get softmax distribution
    let prediction = prediction_tensor.softmax(-1, Kind::Float).get(0);

    if prediction.size()[0] == 0 || prediction.numel() == 0 {
        return Err(CustomError::new("Prediction tensor is empty!"));
    }

    // Get the index of the most likely class
    let class = get_prediction_class(&prediction)?;

    // Get the confidence of the most likely class
    let confidence = prediction.double_value(&[class as i64]);
    Ok(Prediction { class, confidence })
}

#[post("/predict")]
async fn predict(payload: Multipart) -> Result<HttpResponse, CustomError> {
    log::info!("Starting prediction...");

    let tensor = read_image_data_and_preprocess(payload).await?;
    log::info!("Image data read and preprocessed");

    let model = get_model()?;
    log::info!("Model loaded");

    let result = predict_image(tensor, &model).await?;
    log::info!("Image prediction completed");

    println!("Class: {:?}", result.class);
    println!("Confidence: {:?}", result.confidence);
    Ok(HttpResponse::Ok().json(result)).map_err(|e: std::fmt::Error| {
        CustomError::new(&format!("Failed to serialize response: {}", e))
    })
}

/*Ensure PyTorch Bindings For Rust are working properly for cpu */
#[get("/check_pytorch_cpu")]
async fn self_check() -> Result<HttpResponse, Error> {
    let tensor_size = tensor_device_cpu();
    let message = "PyTorch CPU: self check successful with tensor: ".to_string() + &tensor_size;
    log::info!("{}", message);
    Ok(HttpResponse::Ok().json(message))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "actix_web=info");
    env_logger::init();

    HttpServer::new(|| {
        App::new()
            .wrap(Logger::default())
            .service(index)
            .service(predict)
            .service(self_check)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
