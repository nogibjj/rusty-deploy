use actix_multipart::Multipart;
use actix_web::error::ResponseError;
use actix_web::HttpResponse;
use actix_web::{middleware::Logger, post, App, HttpServer, Result};
use futures::StreamExt;
use serde::Serialize;
use std::env;
use std::fmt;
use std::io::Write;
use tch::{Device, IValue, Kind, Tensor};

#[derive(Debug)]
pub enum CustomError {
    ImageError(image::ImageError),
    TchError(tch::TchError),
    ActixMultipartError(actix_multipart::MultipartError),
    IoError(std::io::Error),
}

impl From<std::io::Error> for CustomError {
    fn from(error: std::io::Error) -> Self {
        CustomError::IoError(error)
    }
}

impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CustomError::ImageError(e) => write!(f, "Image error: {}", e),
            CustomError::TchError(e) => write!(f, "Tch error: {}", e),
            CustomError::ActixMultipartError(_) => todo!(),
            CustomError::IoError(_) => todo!(),
        }
    }
}

impl ResponseError for CustomError {
    fn error_response(&self) -> HttpResponse {
        HttpResponse::InternalServerError().json(format!("{}", self))
    }
}

impl From<image::ImageError> for CustomError {
    fn from(e: image::ImageError) -> Self {
        CustomError::ImageError(e)
    }
}

impl From<tch::TchError> for CustomError {
    fn from(e: tch::TchError) -> Self {
        CustomError::TchError(e)
    }
}

impl From<actix_multipart::MultipartError> for CustomError {
    fn from(error: actix_multipart::MultipartError) -> Self {
        CustomError::ActixMultipartError(error)
    }
}

#[derive(Serialize)]
struct Prediction {
    class: usize,
    confidence: f64,
}

#[post("/predict")]
async fn predict(mut payload: Multipart) -> Result<HttpResponse, CustomError> {
    let mut image_data: Vec<u8> = vec![];
    while let Some(item) = payload.next().await {
        let mut field = item?;
        while let Some(chunk) = field.next().await {
            let data = chunk?;
            image_data.write_all(&data)?;
        }
    }

    let image = image::load_from_memory(&image_data)
        .map_err(CustomError::from)?
        .to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);

    let resized_raw = resized.into_raw();
    let norm_img: Vec<f32> = resized_raw.into_iter().map(|v| v as f32 / 255.0).collect();

    let byte_data: Vec<u8> = bytemuck::cast_slice(&norm_img).to_vec();

    let tensor = Tensor::of_data_size(
        &byte_data,
        &[1, 3, 224, 224],
        tch::Kind::Float,
    )
    .permute(&[0, 3, 1, 2])
    .to_device(Device::Cpu);

    let model_path = match env::var("MODEL_PATH") {
        Ok(path) => path,
        Err(_) => "model/resnet34.ot".to_string(),
    };

    let model = tch::CModule::load(model_path).map_err(CustomError::from)?;
    let output = model.forward_is(&[IValue::from(tensor)])?;
    let prediction_tensor = match output {
        IValue::Tensor(tensor) => tensor,
        _ => panic!("Output is not a Tensor!"),
    };

    let prediction = prediction_tensor.softmax(-1, Kind::Float).get(0);
    let class = prediction.argmax(0, false).int64_value(&[0]) as usize;
    let confidence = prediction.double_value(&[class as i64]);
    let result = Prediction { class, confidence };

    Ok(HttpResponse::Ok().json(result))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "actix_web=info");
    env_logger::init();

    HttpServer::new(|| App::new().wrap(Logger::default()).service(predict))
        .bind("127.0.0.1:8080")?
        .run()
        .await
}
