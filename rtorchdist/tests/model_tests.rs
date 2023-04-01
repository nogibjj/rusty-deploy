use rtorchdist::get_prediction_class;
use std::path::Path;
use tch::{IValue, Tensor};
use log::{debug,error};

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

type TestFunction = fn(&tch::CModule) -> Result<(), Box<dyn std::error::Error>>;

fn test_output_tensor_shape(model: &tch::CModule) -> Result<(), Box<dyn std::error::Error>> {
    let input = Tensor::ones(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
    let output = model.forward_is(&[IValue::from(input)])?;
    let prediction_tensor = match output {
        IValue::Tensor(tensor) => tensor,
        _ => panic!("Model output is not a Tensor"),
    };
    assert_eq!(prediction_tensor.size(), &[1, 1000]);
    Ok(())
}

fn test_output_without_get_prediction_class(
    model: &tch::CModule,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = Tensor::ones(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
    let output = model.forward_is(&[IValue::from(input)])?;
    let prediction_tensor = match output {
        IValue::Tensor(tensor) => tensor,
        _ => panic!("Model output is not a Tensor"),
    };
    let max_index = prediction_tensor.argmax(-1, false);
    println!("Max index without get_prediction_class: {:?}", max_index);
    Ok(())
}

fn test_model_output_for_model(model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let model = tch::CModule::load(model_path)?;
    let input = Tensor::ones(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
    let output = model.forward_is(&[IValue::from(input)])?;
    let prediction_tensor = match output {
        IValue::Tensor(tensor) => tensor,
        _ => panic!("Model output is not a Tensor"),
    };
    let class = get_prediction_class(&prediction_tensor)?;
    let confidence = prediction_tensor
        .f_sort_values(&prediction_tensor, &prediction_tensor, class as i64, true)?
        .0;

    println!("Model: {:?}", model_path.file_name());
    println!("Most likely class: {}", class);
    println!("Confidence of most likely class: {}", confidence);
    Ok(())
}

#[test]
fn test_models_no_internal_code() -> Result<(), Box<dyn std::error::Error>> {
    let models = get_models();
    println!("models: {:?}", models);
    let test_functions: Vec<TestFunction> = vec![
        test_output_tensor_shape,
        test_output_without_get_prediction_class,
    ];

    for model_info in models {
        println!("model_info: {:?}", model_info);
        let model = match load_model(&model_info) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        println!("model: {:?}", model);
        for test_function in &test_functions {
            let result = test_function(&model);
            match result {
                Ok(_) => (),
                Err(e) => {
                    println!("test failed: {:?}", e);
                    return Err(e);
                }
            }
        }
    }

    Ok(())
}

#[test]
fn test_model_output_tensor_shape() -> Result<(), Box<dyn std::error::Error>> {
    let models = get_models();

    for model_info in models {
        let model = match load_model(&model_info) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let input = Tensor::ones(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
        let output = model.forward_is(&[IValue::from(input)])?;
        let prediction_tensor = match output {
            IValue::Tensor(tensor) => tensor,
            _ => panic!("Model output is not a Tensor"),
        };

        assert_eq!(prediction_tensor.size(), &[1, 1000]);
    }

    Ok(())
}


#[test]
fn test_load_model() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "model/resnet34.ot";
    let _model = tch::CModule::load(model_path)?;
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


#[test]
fn test_model_output() -> Result<(), Box<dyn std::error::Error>> {
    let model_paths = [
        "model/resnet18.ot",
        "model/resnet34.ot",
        "model/densenet121.ot",
        "model/vgg13.ot",
        "model/vgg16.ot",
        "model/vgg19.ot",
        "model/squeezenet1_0.ot",
        "model/squeezenet1_1.ot",
        "model/alexnet.ot",
        "model/inception-v3.ot",
        "model/mobilenet-v2.ot",
    ];

    for model_path in &model_paths {
        let path = Path::new(model_path);
        test_model_output_for_model(&path)?;
    }

    Ok(())
}

#[test]
fn test_get_prediction_class() -> Result<(), Box<dyn std::error::Error>> {
    let prediction_tensor = Tensor::of_slice(&[0.1, 0.2, 0.7]);
    assert_eq!(get_prediction_class(&prediction_tensor)?, 2);

    let prediction_tensor = Tensor::of_slice(&[0.33, 0.33, 0.34]);
    assert_eq!(get_prediction_class(&prediction_tensor)?, 2);

    let prediction_tensor = Tensor::of_slice(&[0.0, 0.0, 1.0]);
    assert_eq!(get_prediction_class(&prediction_tensor)?, 2);

    let prediction_tensor = Tensor::of_slice(&[0.5, 0.3, 0.2]);
    assert_eq!(get_prediction_class(&prediction_tensor)?, 0);
    Ok(())
}

#[test]
fn test_models() -> Result<(), Box<dyn std::error::Error>> {
    let models = get_models();
    debug!("models: {:?}", models);
    let test_functions: Vec<TestFunction> = vec![
        test_output_tensor_shape,
        test_output_without_get_prediction_class,
    ];

    for model_info in models {
        debug!("model_info: {:?}", model_info);
        let model = match load_model(&model_info) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        debug!("model: {:?}", model);
        for test_function in &test_functions {
            let result = test_function(&model);
            match result {
                Ok(_) => (),
                Err(e) => {
                    error!("test failed: {:?}", e);
                    return Err(e);
                }
            }
        }
    }

    Ok(())
}
