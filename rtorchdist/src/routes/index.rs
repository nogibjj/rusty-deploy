use actix_web::{get, HttpResponse};

#[get("/")]
pub async fn index() -> HttpResponse {
    log::info!("route: / function: index()");
    let message = "Send an image payload using curl with the following command:\ncurl -X POST -H \"Content-Type: multipart/form-data\" -F \"image=@/path/to/your/image.jpg\" http://127.0.0.1:8080/predict";
    HttpResponse::Ok().content_type("text/plain").body(message)
}

pub fn config(cfg: &mut actix_web::web::ServiceConfig) {
    cfg.service(index);
}
