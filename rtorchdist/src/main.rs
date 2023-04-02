use actix_web::middleware::Logger;
use actix_web::{App, HttpServer};

mod logic;
mod routes;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    println!("Starting pytorch model server...");
    HttpServer::new(|| {
        App::new()
            .wrap(Logger::default())
            .service(routes::index)
            .service(routes::check_image_prediction)
            .service(routes::check_image_upload)
            .service(routes::check_pytorch_cpu)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
