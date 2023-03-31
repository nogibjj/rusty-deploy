use rtorchdist::get_prediction_class;
use tch::{IValue, Tensor};

#[test]
fn test_load_model() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "model/resnet34.ot";
    let  _model = tch::CModule::load(model_path)?;
    Ok(())
}

#[test]
fn test_get_prediction_class_no_model() -> Result<(), Box<dyn std::error::Error>> {
    let output = Tensor::of_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(get_prediction_class(&output)?, 2);
    Ok(())
}

#[test]
fn test_model_output() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "model/resnet34.ot";
    let model = tch::CModule::load(model_path)?;
    let input = Tensor::ones(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
    let output = model.forward_is(&[IValue::from(input)])?;
    let prediction_tensor = match output {
        IValue::Tensor(tensor) => tensor,
        _ => panic!("Model output is not a Tensor"),
    };
    let class = get_prediction_class(&prediction_tensor)?;
    let confidence = prediction_tensor.double_value(&[class as i64]);
    println!("Most likely class: {}", class);
    println!("Confidence of most likely class: {}", confidence);
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
