//use actix_multipart::Multipart;
//use actix_web::post;
use actix_web::get;

use actix_web::{middleware::Logger, App, HttpServer};
use actix_web::{Error, HttpResponse, Result};
use log::info;
use serde_json::json;

//from lib
use rtorchdist::{self_check_predict, tensor_device_cpu};

#[get("/")]
async fn index() -> HttpResponse {
    let message = "Send an image payload using curl with the following command:\ncurl -X POST -H \"Content-Type: multipart/form-data\" -F \"image=@/path/to/your/image.jpg\" http://127.0.0.1:8080/predict";
    HttpResponse::Ok().content_type("text/plain").body(message)
}
/*
#[post("/predict")]
async fn predict(payload: Multipart) -> Result<HttpResponse, CustomError> {
    log::info!("Starting prediction...");

    // Get the image data from the payload using preprocess_image first convert to Vec<u8>
    // convert payload to Vec<u8>

    log::info!("Image data read and preprocessed");
    let _payload = convert_payload_to_vec_u8(payload).await?;
    log::info!("Model loaded");
    log::info!("Image prediction completed");
    let result = "Image prediction completed";
    Ok(HttpResponse::Ok().json(result)).map_err(|e: std::fmt::Error| {
        CustomError::new(&format!("Failed to serialize response: {}", e))
    })
}
*/
/*A self check that loads a local model and image with known results
uses: self_check_predict() from lib.rs
 */
#[get("/check_image_prediction")]
async fn self_check_image_predict() -> HttpResponse {
    match self_check_predict() {
        Ok(result) => {
            log::info!(
                "Route: /check_image_prediction, Function: self_check_image_predict, Result: {:?}",
                result
            );
            HttpResponse::Ok().json(json!({"status": "success", "result": result}))
        }
        Err(e) => {
            let error_message = format!("Image prediction self check failed with error: {:?}", e);
            log::error!(
                "Route: /check_image_prediction, Function: self_check_image_predict, Error: {}",
                error_message
            );
            HttpResponse::InternalServerError()
                .json(json!({"status": "error", "message": error_message}))
        }
    }
}

/*Ensure PyTorch Bindings For Rust are working properly for cpu */
#[get("/check_pytorch_cpu")]
async fn self_check() -> Result<HttpResponse, Error> {
    let tensor_size = tensor_device_cpu();
    let message = "PyTorch CPU: self check successful with tensor: ".to_string() + &tensor_size;
    //include route and function name in log message i.e. /check_pytorch_cpu/self_check
    info!(
        "route: check_pytorch_cpu: function: self_check(){:?}",
        message
    );
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
            .service(self_check_image_predict)
            //.service(predict)
            .service(self_check)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
