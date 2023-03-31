use actix_multipart::Multipart;
use actix_web::error::ErrorInternalServerError;
use actix_web::error::ResponseError;
use actix_web::get;
use actix_web::{middleware::Logger, post, App, HttpServer};
use actix_web::{Error, HttpResponse, Result};
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
    Other(String),
}

impl CustomError {
    fn new(message: &str) -> Self {
        CustomError::Other(message.to_string())
    }
}

impl From<Error> for CustomError {
    fn from(error: Error) -> Self {
        CustomError::new(&error.to_string())
    }
}

impl From<std::io::Error> for CustomError {
    fn from(error: std::io::Error) -> Self {
        CustomError::IoError(error)
    }
}

// Update the Display implementation for CustomError
impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CustomError::ImageError(e) => write!(f, "ImageError: {}", e),
            CustomError::TchError(e) => write!(f, "TchError: {}", e),
            CustomError::ActixMultipartError(e) => write!(f, "ActixMultipartError: {}", e),
            CustomError::IoError(e) => write!(f, "IoError: {}", e),
            // Add a match arm for the Other variant
            CustomError::Other(e) => write!(f, "OtherError: {}", e),
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

#[get("/")]
async fn index() -> HttpResponse {
    HttpResponse::Ok().content_type("text/plain").body("Send an image payload using curl with the following command:\ncurl -X POST -H \"Content-Type: multipart/form-data\" -F \"image=@/path/to/your/image.jpg\" http://127.0.0.1:8080/predict")
}

async fn read_image_data(mut payload: Multipart) -> Result<Vec<u8>, CustomError> {
    let mut image_data = vec![];
    while let Some(item) = payload.next().await {
        let mut field = item?;
        while let Some(chunk) = field.next().await {
            let data = chunk?;
            image_data.write_all(&data)?;
        }
    }
    Ok(image_data)
}

fn preprocess_image(image_data: Vec<u8>) -> Result<Tensor, CustomError> {
    println!("Received image data with length {}", image_data.len());
    let image = image::load_from_memory(&image_data)
        .map_err(CustomError::from)?
        .to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);

    let resized_raw = resized.into_raw();
    let norm_img: Vec<f32> = resized_raw.into_iter().map(|v| v as f32 / 255.0).collect();
    println!("norm_img: {:?}", norm_img);
    let byte_data: Vec<u8> = bytemuck::cast_slice(&norm_img).to_vec();

    let tensor = Tensor::of_data_size(&byte_data, &[1, 224, 224, 3], tch::Kind::Float)
        .permute(&[0, 3, 1, 2])
        .to_device(Device::Cpu);
    println!("Created tensor with size {:?}", tensor.size());
    println!("Tensor:\n{}", tensor);
    Ok(tensor)
}

fn get_model() -> Result<tch::CModule, CustomError> {
    let model_path = match env::var("MODEL_PATH") {
        Ok(path) => path,
        Err(_) => "model/resnet34.ot".to_string(),
    };

    tch::CModule::load(model_path).map_err(CustomError::from)
}

fn get_prediction_class(prediction: &Tensor) -> Result<usize, Box<dyn std::error::Error>> {
    let class = prediction.argmax(0, false).int64_value(&[0]) as usize;
    Ok(class)
}

#[post("/predict")]
async fn predict(payload: Multipart) -> Result<HttpResponse, CustomError> {
    // Read image data from multipart request
    let image_data = read_image_data(payload).await?;
    // Preprocess image data into a Tensor
    let tensor = preprocess_image(image_data)?;
    // Load the model
    let model = get_model()?;
    // Move tensor to CPU
    let input_tensor = tensor.to_device(Device::Cpu);

    println!("Input tensor size: {:?}", input_tensor.size());
    println!("Input tensor dtype: {:?}", input_tensor.kind());
    // Forward pass through model
    let output = model.forward_is(&[IValue::from(input_tensor)])?;

    let prediction_tensor = match output {
        IValue::Tensor(tensor) => tensor,
        _ => return Err(CustomError::new("Output is not a Tensor!")),
    };

    // Get softmax distribution
    let prediction = prediction_tensor.softmax(-1, Kind::Float).get(0);
    println!("Output tensor size: {:?}", prediction.size());
    println!("Output tensor dtype: {:?}", prediction.kind());

    if prediction.size()[0] == 0 {
        return Err(CustomError::new("Prediction tensor is empty!"));
    }

    if prediction.numel() == 0 {
        return Err(CustomError::new("Prediction tensor is empty!"));
    }

    // Get the index of the most likely class
    let class_option = prediction.argmax(0, false);

    // Handle the case when `int64_value` returns `None`
    if let Some(value) = class_option.int64_value(&[0]) {
        let class = value as usize;

        // Get the confidence of the most likely class
        let confidence = prediction.double_value(&[class as i64]);
        let result = Prediction { class, confidence };

        println!("Class: {:?}", class);
        println!("Confidence: {:?}", confidence);
        Ok(HttpResponse::Ok().json(result))
    } else {
        Err(CustomError::new(
            "Failed to get class from prediction tensor",
        ))
    }
}

#[get("/self_check")]
async fn self_check() -> Result<HttpResponse, Error> {
    let model_path = env::var("MODEL_PATH").unwrap_or_else(|_| "model/resnet34.ot".to_string());

    let model = tch::CModule::load(model_path)
        .map_err(|e| ErrorInternalServerError(format!("Failed to load model: {}", e)))?;

    let dummy_input = Tensor::ones(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
    let output = model
        .forward_is(&[IValue::from(dummy_input)])
        .map_err(|e| ErrorInternalServerError(format!("Model failed on input: {}", e)))?;

    let _prediction_tensor = match output {
        IValue::Tensor(tensor) => tensor,
        _ => {
            return Err(ErrorInternalServerError(
                "Model output is not a Tensor".to_string(),
            ));
        }
    };

    Ok(HttpResponse::Ok().json("Self-check completed successfully!"))
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
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
