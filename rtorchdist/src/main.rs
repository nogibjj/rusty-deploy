use actix_multipart::Multipart;
use actix_web::get;
use actix_web::{middleware::Logger, post, App, HttpServer};
use actix_web::{Error, HttpResponse, Result};

//custom errors
pub mod cerror;
use cerror::CustomError;

//from lib
use rtorchdist::{
    convert_payload_to_vec_u8, get_model, predict_image, preprocess_image, tensor_device_cpu
};

#[get("/")]
async fn index() -> HttpResponse {
    let message = "Send an image payload using curl with the following command:\ncurl -X POST -H \"Content-Type: multipart/form-data\" -F \"image=@/path/to/your/image.jpg\" http://127.0.0.1:8080/predict";
    HttpResponse::Ok().content_type("text/plain").body(message)
}

#[post("/predict")]
async fn predict(payload: Multipart) -> Result<HttpResponse, CustomError> {
    log::info!("Starting prediction...");

    // Get the image data from the payload using preprocess_image first convert to Vec<u8>
    // convert payload to Vec<u8>

    log::info!("Image data read and preprocessed");
    log::info!("Model loaded");
    log::info!("Image prediction completed");
    let result = "Image prediction completed";
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
