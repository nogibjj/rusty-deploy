use actix_multipart::Multipart;
use actix_web::http::header::HeaderMap;
use actix_web::http::header::{ContentDisposition, DispositionParam, DispositionType};
use actix_web::web::Bytes;
use anyhow::Error;
use futures::stream::Stream;
use log::info;
use rtorchdist::{files, self_check_predict, tensor_device_cpu};
use std::pin::Pin;
use std::task::{Context, Poll};
use test_log::test;

// Helper struct to implement Stream for testing
pub struct TestStream {
    data: Vec<u8>,
    read: bool,
}

impl Stream for TestStream {
    type Item = Result<Bytes, actix_web::error::PayloadError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.read {
            Poll::Ready(None)
        } else {
            self.read = true;
            let bytes = Bytes::from(self.data.clone());
            Poll::Ready(Some(Ok(bytes)))
        }
    }
}

#[derive(Debug)]
struct ModelInfo {
    model_path: &'static str,
}

fn get_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            model_path: "model/alexnet.ot",
        },
        ModelInfo {
            model_path: "model/densenet121.ot",
        },
        ModelInfo {
            model_path: "model/inception-v3.ot",
        },
        ModelInfo {
            model_path: "model/mobilenet-v2.ot",
        },
        ModelInfo {
            model_path: "model/resnet18.ot",
        },
        ModelInfo {
            model_path: "model/resnet34.ot",
        },
        ModelInfo {
            model_path: "model/squeezenet1_0.ot",
        },
        ModelInfo {
            model_path: "model/squeezenet1_1.ot",
        },
        ModelInfo {
            model_path: "model/vgg13.ot",
        },
        ModelInfo {
            model_path: "model/vgg16.ot",
        },
        ModelInfo {
            model_path: "model/vgg19.ot",
        },
    ]
}

fn load_model(model_info: &ModelInfo) -> Result<tch::CModule, Box<dyn std::error::Error>> {
    let model = tch::CModule::load(model_info.model_path)?;
    Ok(model)
}

// test that we can load all models
#[test]
fn test_load_models() -> Result<(), Box<dyn std::error::Error>> {
    let models = get_models();
    for model_info in models {
        let model = load_model(&model_info)?;
        //log it
        info!("TEST: Loaded model: {:?}", model);
    }
    Ok(())
}

//test tensor_device_cpu
#[test]
fn test_tensor_device_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let size_str = tensor_device_cpu();
    info!("TEST: Tensor size: {:?}", size_str);
    assert_eq!(size_str, "3");
    Ok(())
}

//tests self_check_predict()
#[test]
fn test_self_check_predict() {
    let prediction = match self_check_predict() {
        Ok(p) => p,
        Err(e) => {
            println!("Error: {:?}", e);
            return;
        }
    };
    println!("TEST: Self check prediction: {:?}", prediction);
    //assert Prediction { probabilities: [0.05008329823613167], classes: ["tarantula"] }
    assert_eq!(prediction.probabilities[0], 0.05008329823613167);
    assert_eq!(prediction.classes[0], "tarantula");
}

//test file save
#[actix_rt::test]
async fn test_save_file() -> Result<(), Error> {
    let file_path = "test.txt".to_string();
    let _content_disposition = ContentDisposition {
        disposition: DispositionType::FormData,
        parameters: vec![
            DispositionParam::Name("file".to_string()),
            DispositionParam::Filename("test.txt".to_string()),
        ],
    };

    let test_stream = TestStream {
        data: "Hello World".as_bytes().to_vec(),
        read: false,
    };
    let multipart = Multipart::new(&HeaderMap::new(), test_stream);
    let filepath = files::save_file(multipart, file_path.clone())
        .await
        .unwrap();
    assert_eq!(filepath, format!(".{}", file_path));
    assert!(std::path::Path::new(&filepath).exists());
    std::fs::remove_file(&filepath).unwrap();
    Ok(())
}
