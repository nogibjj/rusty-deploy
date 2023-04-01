use log::info;
use rtorchdist::{get_model, predict_image, preprocess_image, tensor_device_cpu, Prediction};
use std::fs::File;
use std::io::Read;
use test_env_log::test;



#[derive(Debug)]
struct ModelInfo {
    model_path: &'static str,
}

fn read_image_data(file_path: &str) -> Vec<u8> {
    let mut file = File::open(file_path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    buffer
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
        info!("Loaded model: {:?}", model);
    }
    Ok(())
}

#[test]
fn test_preprocess_image() {
    let image_data = read_image_data("tests/fixtures/lion.jpg");
    let tensor = preprocess_image(image_data);
    assert!(tensor.is_ok());
}

//test predict_image
#[test]
fn test_predict_image() -> Result<(), Box<dyn std::error::Error>> {
    info!("TEST: load image data");
    let image_data = read_image_data("tests/fixtures/lion.jpg");
    info!("TEST: load model");
    let model = get_model()?;
    //convert image to tensor
    info!("preprocess image");
    let image_data = preprocess_image(image_data)?;
    info!("predict image");
    let prediction = predict_image(image_data, &model)?;
    //log it
    info!("Prediction: {:?}", prediction);
    Ok(())
}

//test tensor_device_cpu
#[test]
fn test_tensor_device_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let size_str = tensor_device_cpu();
    println!("Tensor size: {:?}", size_str);
    assert_eq!(size_str, "3");
    Ok(())
}
