use log::{debug, error};
use rtorchdist::get_prediction_class;
use std::path::Path;
use tch::{IValue, Tensor};

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
        debug!("Loaded model: {:?}", model);
    }
    Ok(())
}

#[test]
fn test_get_prediction_class_no_model() -> Result<(), Box<dyn std::error::Error>> {
    let output = Tensor::of_slice(&[1.0, 2.0, 3.0]).unsqueeze(0);
    println!("Output tensor: {:?}", output);

    let class = get_prediction_class(&output)?;
    println!("Class: {:?}", class);

    assert_eq!(class, 2);
    Ok(())
}
