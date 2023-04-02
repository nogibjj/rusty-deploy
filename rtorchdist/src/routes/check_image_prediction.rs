use actix_web::{get, HttpResponse};
use rtorchdist::self_check_predict;
use serde_json::json;

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

pub fn config(cfg: &mut actix_web::web::ServiceConfig) {
    cfg.service(check_image_prediction);
}
