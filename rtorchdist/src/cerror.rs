use actix_multipart::MultipartError;
use actix_web::Error as ActixWebError;
use actix_web::ResponseError;
use image::ImageError;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use tch::TchError;

#[derive(Debug, Serialize, Deserialize)]
pub struct CustomError {
    message: String,
}

impl From<ActixWebError> for CustomError {
    fn from(error: ActixWebError) -> Self {
        CustomError::new(&error.to_string())
    }
}

impl CustomError {
    pub fn new(message: &str) -> Self {
        CustomError {
            message: message.to_string(),
        }
    }
}

impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for CustomError {}

impl From<&str> for CustomError {
    fn from(error: &str) -> Self {
        CustomError::new(error)
    }
}

impl From<std::io::Error> for CustomError {
    fn from(error: std::io::Error) -> Self {
        CustomError::new(&error.to_string())
    }
}

impl From<MultipartError> for CustomError {
    fn from(error: MultipartError) -> Self {
        CustomError::new(&error.to_string())
    }
}

impl From<ImageError> for CustomError {
    fn from(error: ImageError) -> Self {
        CustomError::new(&error.to_string())
    }
}

impl From<TchError> for CustomError {
    fn from(error: TchError) -> Self {
        CustomError::new(&error.to_string())
    }
}

impl ResponseError for CustomError {
    fn error_response(&self) -> actix_web::HttpResponse {
        actix_web::HttpResponse::InternalServerError().json(self)
    }
}
