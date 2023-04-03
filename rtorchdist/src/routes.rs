use actix_multipart::Multipart;
use actix_web::post;
use actix_web::{get, Error, HttpResponse, Result};
use serde_json::json;
use std::path::Path;

use crate::logic::files;
use crate::logic::predict_image;
use crate::logic::self_check_predict;
use crate::logic::tensor_device_cpu;

#[get("/")]
pub async fn index() -> HttpResponse {
    log::info!("route: / function: index()");
    let message = "Send an image payload using curl with the following command:\ncurl -X POST -H \"Content-Type: multipart/form-data\" -F \"image=@/path/to/your/image.jpg\" http://127.0.0.1:8080/predict";
    HttpResponse::Ok().content_type("text/plain").body(message)
}

#[post("/predict")]
pub async fn predict(payload: Multipart) -> Result<HttpResponse, Error> {
    //log starting upload and include route and function name
    log::info!("route: /predict function: predict()");
    let file_path = "/tmp/check_image_upload".to_string();
    let file_path = match files::save_file(payload, file_path).await {
        Ok(path) => path,
        Err(e) => {
            let error_message = format!("File upload failed with error: {:?}", e);
            log::error!(
                "Route: /predict, Function: predict_image, Error: {}",
                error_message
            );
            return Ok(
                HttpResponse::Ok().json(json!({ "status": "error", "message": error_message }))
            );
        }
    };
    let cloned_file_path = file_path.clone();
    let prediction = match predict_image(cloned_file_path).await {
        Ok(p) => p,
        Err(e) => {
            let error_message = format!("Prediction failed with error: {:?}", e);
            log::error!(
                "Route: /predict, Function: predict_image, Error: {}",
                error_message
            );
            return Ok(HttpResponse::InternalServerError()
                .json(json!({ "status": "error", "message": error_message })));
        }
    };
    //delete file after prediction
    std::fs::remove_file(file_path)?;
    log::info!(
        "Route: /predict, Function: predict_image, Result: {:?}",
        prediction
    );

    Ok(HttpResponse::Ok().json(json!({ "status": "success", "result": prediction })))
}

#[post("/check_image_upload")]
pub async fn check_image_upload(payload: Multipart) -> Result<HttpResponse, Error> {
    // log starting upload and include route and function name
    log::info!("route: /check_image_upload function: check_image_upload()");

    // create the directory if it doesn't exist
    let dir = Path::new("./tmp/check_image_upload");
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }

    // use pattern matching to handle the result of saving the file
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

    // return a successful response with the result
    Ok(HttpResponse::Ok().json(result))
}

#[get("/check_pytorch_cpu")]
async fn check_pytorch_cpu() -> Result<HttpResponse, Error> {
    let tensor_size = tensor_device_cpu();
    let message = "PyTorch CPU: self check successful with tensor: ".to_string() + &tensor_size;
    log::info!("route: /check_pytorch_cpu function: self_check()");
    Ok(HttpResponse::Ok().json(message))
}

#[get("/check_image_prediction")]
pub async fn check_image_prediction() -> HttpResponse {
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
