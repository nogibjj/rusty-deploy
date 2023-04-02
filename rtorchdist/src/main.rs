use actix_multipart::Multipart;
use actix_web::{get, post};

use actix_web::{middleware::Logger, App, HttpServer};
use actix_web::{Error, HttpResponse, Result};
use serde_json::json;

use std::path::Path;

//from lib
use rtorchdist::{files, self_check_predict, tensor_device_cpu};

#[get("/")]
async fn index() -> HttpResponse {
    log::info!("route: / function: index()");
    let message = "Send an image payload using curl with the following command:\ncurl -X POST -H \"Content-Type: multipart/form-data\" -F \"image=@/path/to/your/image.jpg\" http://127.0.0.1:8080/predict";
    HttpResponse::Ok().content_type("text/plain").body(message)
}
/*
#[post("/predict")]
async fn predict(payload: Multipart) -> Result<HttpResponse, CustomError> {
    log::log::info!("Starting prediction...");

    // Get the image data from the payload using preprocess_image first convert to Vec<u8>
    // convert payload to Vec<u8>

    log::log::info!("Image data read and preprocessed");
    let _payload = convert_payload_to_vec_u8(payload).await?;
    log::log::info!("Model loaded");
    log::log::info!("Image prediction completed");
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
    log::info!(
        "route: check_pytorch_cpu: function: self_check(){:?}",
        message
    );
    Ok(HttpResponse::Ok().json(message))
}

#[post("/check_image_upload")]
async fn check_image_upload(payload: Multipart) -> Result<HttpResponse, Error> {
    //log starting upload and include route and function name
    log::info!("route: /check_image_upload function: check_image_upload()");

    let dir = Path::new("./tmp/check_image_upload");
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }

    let result = match files::save_file(payload, "/tmp/check_image_upload".to_string()).await {
        Ok(path) => {
            let status = "success".to_string();
            json!({ "status": status, "filepath": path })
        }
        Err(e) => {
            let status = "error".to_string();
            let error = format!("{:?}", e);
            json!({ "status": status, "error": error })
        }
    };

    Ok(HttpResponse::Ok().json(result))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    //print server info on start
    println!("Starting pytorch model server...");
    HttpServer::new(|| {
        App::new()
            .wrap(Logger::default())
            .service(index)
            .service(self_check_image_predict)
            .service(check_image_upload)
            .service(self_check)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
