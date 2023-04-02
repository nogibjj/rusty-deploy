use actix_multipart::Multipart;
use actix_web::post;
use actix_web::{Error, HttpResponse};
use serde_json::json;
use std::path::Path;

// Import the functions from the `files` module
use rtorchdist::files;

#[post("/check_image_upload")]
pub async fn check_image_upload(payload: Multipart) -> Result<HttpResponse, Error> {
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

pub fn config(cfg: &mut actix_web::web::ServiceConfig) {
    cfg.service(check_image_upload); // Change this line
}
