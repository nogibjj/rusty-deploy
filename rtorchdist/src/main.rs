use actix_web::middleware::Logger;
use actix_web::{App, HttpServer};

mod routes;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    println!("Starting pytorch model server...");
    HttpServer::new(|| {
        App::new()
            .wrap(Logger::default())
            .configure(routes::index::config)
            .configure(routes::check_image_prediction::config)
            .configure(routes::check_image_upload::config)
            .configure(routes::check_pytorch_cpu::config)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
