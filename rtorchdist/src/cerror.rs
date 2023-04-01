use actix_multipart::MultipartError;
use actix_web::{HttpResponse, ResponseError};
use anyhow::anyhow;
use image::ImageError;
use std::error::Error;
use std::fmt;
use std::num::ParseFloatError;
use tch::TchError;
use tch::TchError as TensorError;

#[derive(Debug)]
pub enum PredictError {
    InvalidInput,
    InvalidModel,
    RuntimeError,
}

impl PredictError {
    pub fn new_invalid_input() -> PredictError {
        PredictError::InvalidInput
    }

    pub fn new_invalid_model() -> PredictError {
        PredictError::InvalidModel
    }

    pub fn new_runtime_error() -> PredictError {
        PredictError::RuntimeError
    }
}

impl fmt::Display for PredictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            PredictError::InvalidInput => "Invalid input",
            PredictError::InvalidModel => "Invalid model",
            PredictError::RuntimeError => "Runtime error",
        };
        write!(f, "{}", message)
    }
}

impl From<PredictError> for anyhow::Error {
    fn from(error: PredictError) -> Self {
        match error {
            PredictError::InvalidInput => anyhow!("Invalid input"),
            PredictError::InvalidModel => anyhow!("Invalid model"),
            PredictError::RuntimeError => anyhow!("Runtime error"),
        }
    }
}

#[derive(Debug)]
pub enum CustomError {
    ImageError(ImageError),
    PayloadError(String),
    MultipartError(MultipartError),
    ParseFloatError(ParseFloatError),
    TchError(TchError),
    TensorError(TensorError),
    UnknownError(String),
}

impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CustomError::ImageError(e) => write!(f, "ImageError: {}", e),
            CustomError::PayloadError(e) => write!(f, "PayloadError: {}", e),
            CustomError::MultipartError(e) => write!(f, "MultipartError: {}", e),
            CustomError::ParseFloatError(e) => write!(f, "ParseFloatError: {}", e),
            CustomError::TchError(e) => write!(f, "TchError: {}", e),
            CustomError::TensorError(e) => write!(f, "TensorError: {}", e),
            CustomError::UnknownError(e) => write!(f, "UnknownError: {}", e),
        }
    }
}

impl CustomError {
    pub fn new(msg: &str) -> CustomError {
        CustomError::PayloadError(msg.to_string())
    }
}

impl Error for CustomError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CustomError::ImageError(e) => Some(e),
            CustomError::PayloadError(_) => None,
            CustomError::MultipartError(e) => Some(e),
            CustomError::ParseFloatError(e) => Some(e),
            CustomError::TchError(e) => Some(e),
            CustomError::TensorError(e) => Some(e),
            CustomError::UnknownError(_) => None,
        }
    }
}

impl From<ImageError> for CustomError {
    fn from(error: ImageError) -> Self {
        CustomError::ImageError(error)
    }
}

impl From<ParseFloatError> for CustomError {
    fn from(error: ParseFloatError) -> Self {
        CustomError::ParseFloatError(error)
    }
}

impl From<actix_multipart::MultipartError> for CustomError {
    fn from(error: actix_multipart::MultipartError) -> Self {
        CustomError::MultipartError(error)
    }
}

impl From<TchError> for CustomError {
    fn from(error: TchError) -> Self {
        CustomError::TchError(error)
    }
}

impl ResponseError for CustomError {
    fn error_response(&self) -> HttpResponse {
        match self {
            CustomError::ImageError(_) => HttpResponse::BadRequest().body("Bad image data"),
            CustomError::PayloadError(e) => HttpResponse::BadRequest().body(e.to_owned()),
            CustomError::MultipartError(e) => HttpResponse::BadRequest().body(e.to_string()),
            CustomError::ParseFloatError(_) => HttpResponse::BadRequest().body("Bad input data"),
            CustomError::TchError(_) => HttpResponse::InternalServerError().finish(),
            CustomError::TensorError(_) => HttpResponse::InternalServerError().finish(),
            CustomError::UnknownError(e) => HttpResponse::InternalServerError().body(e.to_owned()),
        }
    }
}

impl From<std::io::Error> for CustomError {
    fn from(error: std::io::Error) -> Self {
        CustomError::PayloadError(error.to_string())
    }
}

impl From<PredictError> for CustomError {
    fn from(error: PredictError) -> Self {
        CustomError::UnknownError(format!("{:?}", error))
    }
}

impl From<PredictError> for TchError {
    fn from(error: PredictError) -> Self {
        TchError::Kind(error.to_string())
    }
}
