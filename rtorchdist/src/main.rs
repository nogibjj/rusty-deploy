use actix_multipart::{Field, Multipart};
use actix_web::error::ErrorInternalServerError;
use actix_web::error::ResponseError;
use actix_web::get;
use actix_web::http::header::ContentDisposition;
use actix_web::{middleware::Logger, post, App, HttpServer};
use actix_web::{Error, HttpResponse, Result};
use futures::StreamExt;
use serde::Serialize;
use std::env;
use std::fmt;
use tch::{Device, IValue, Kind, Tensor};

//from lib
use rtorchdist::get_prediction_class;

#[derive(Debug)]
pub enum CustomError {
    ImageError(image::ImageError),
    TchError { field1: tch::TchError },
    ActixMultipartError(actix_multipart::MultipartError),
    IoError(std::io::Error),
    Other(String),
    NewVariant(u32),
    StringError(String),
    Message(String),
}

// Add TensorError to the trait definition
trait IntoCustomError<T, TensorError> {
    fn into_custom_error(self, message: &str) -> Result<T, TensorError>;
}

// Implement the trait for Tensor
impl IntoCustomError<tch::Tensor, CustomError> for tch::Tensor {
    fn into_custom_error(self, _message: &str) -> Result<tch::Tensor, CustomError> {
        Ok(self)
    }
}

impl<T> IntoCustomError<T, CustomError> for Result<T, String> {
    fn into_custom_error(self, message: &str) -> Result<T, CustomError> {
        self.map_err(|e| CustomError::StringError(format!("{}: {}", message, e)))
    }
}

// Implement From<&str> for CustomError
impl From<&str> for CustomError {
    fn from(message: &str) -> Self {
        CustomError::Message(message.to_string())
    }
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

impl From<Box<dyn std::error::Error>> for CustomError {
    fn from(error: Box<dyn std::error::Error>) -> Self {
        CustomError::Other(error.to_string())
    }
}

// Update the Display implementation for CustomError
impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CustomError::ImageError(e) => write!(f, "ImageError: {}", e),
            CustomError::TchError { field1: e } => write!(f, "TchError: {}", e),
            CustomError::ActixMultipartError(e) => write!(f, "ActixMultipartError: {}", e),
            CustomError::IoError(e) => write!(f, "IoError: {}", e),
            // Add a match arm for the Other variant
            CustomError::Other(e) => write!(f, "OtherError: {}", e),
            CustomError::NewVariant(e) => write!(f, "NewVariantError: {}", e),
            CustomError::StringError(ref s) => write!(f, "StringError: {}", s),
            CustomError::Message(s) => write!(f, "Message: {}", s),
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
        CustomError::TchError { field1: e }
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

#[get("/self_check")]
async fn self_check() -> Result<HttpResponse, Error> {
    //add log message that describes the function
    log::info!("Starting self check in function: {}", module_path!());
    //log the route that is being called
    log::info!("Calling route: {}", module_path!());

    let model_path = env::var("MODEL_PATH").unwrap_or_else(|_| "model/resnet34.ot".to_string());
    log::info!("Model path: {}", model_path);

    let model = tch::CModule::load(&model_path)
        .map_err(|e| ErrorInternalServerError(format!("Failed to load model: {}", e)))?;
    log::info!("Model loaded successfully");

    let dummy_input = Tensor::ones(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
    log::info!("Dummy input tensor:\n{}", dummy_input);

    let output = model
        .forward_is(&[IValue::from(dummy_input)])
        .map_err(|e| ErrorInternalServerError(format!("Model failed on input: {}", e)))?;
    log::info!("Forward pass through model completed successfully");

    let prediction_tensor = match output {
        IValue::Tensor(tensor) => tensor,
        _ => {
            return Err(ErrorInternalServerError(
                "Model output is not a Tensor".to_string(),
            ));
        }
    };
    log::info!("Prediction tensor:\n{}", prediction_tensor);

    let prediction_size = prediction_tensor.size();
    if prediction_size[0] == 0 || prediction_tensor.numel() == 0 {
        return Err(ErrorInternalServerError(
            "Prediction tensor is empty!".to_string(),
        ));
    }
    log::info!("Prediction tensor size: {:?}", prediction_size);

    let class = get_prediction_class(&prediction_tensor)?;
    log::info!("Most likely class index: {:?}", class);

    let confidence = prediction_tensor.double_value(&[class as i64]);
    log::info!("Confidence of most likely class: {}", confidence);

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
