use actix_web::{get, Error, HttpResponse, Result};

use rtorchdist::tensor_device_cpu;

#[get("/check_pytorch_cpu")]
async fn check_pytorch_cpu() -> Result<HttpResponse, Error> {
    let tensor_size = tensor_device_cpu();
    let message = "PyTorch CPU: self check successful with tensor: ".to_string() + &tensor_size;
    log::info!("route: /check_pytorch_cpu function: self_check()");
    Ok(HttpResponse::Ok().json(message))
}

pub fn config(cfg: &mut actix_web::web::ServiceConfig) {
    cfg.service(check_pytorch_cpu);
}
