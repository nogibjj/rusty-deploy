use log::info;
use rtorchdist::{self_check_predict, tensor_device_cpu};
use std::fs::File;
use std::io::Read;
use test_log::test;

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
    //assert that the word "Persian cat" is in the prediction classes
    assert!(prediction.classes.contains(&"Persian cat".to_string()));
}
