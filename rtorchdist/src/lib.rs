use log::{error, info};
use serde::{Deserialize, Serialize};
use std::env;
use tch::nn::ModuleT;
use tch::vision::imagenet;
use tch::Kind;
use tch::{Device, Tensor};

#[derive(Serialize, Deserialize, Debug)]
pub struct Prediction {
    pub probabilities: Vec<f64>,
    pub classes: Vec<String>,
}

/*simple selfcheck:  has logging at each step
format module_name::function_name
*/
pub fn self_check_predict() -> Result<Prediction, Box<dyn std::error::Error>> {
    info!("func: self_check_predict: starting");
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    info!("func: self_check_predict: loading image: tests/fixtures/lion.jpg");
    let image = imagenet::load_image_and_resize224("tests/fixtures/lion.jpg").unwrap();
    info!("func: self_check_predict: loading model: model/resnet34.ot");
    let weight_file = match env::var("MODEL_PATH") {
        Ok(path) => path,
        Err(_) => "model/resnet34.ot".to_string(),
    };
    let resnet34 = tch::vision::resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT);
    vs.load(weight_file).unwrap();
    // Apply the forward pass of the model to get the logits and convert them
    // to probabilities via a softmax.
    info!("func: self_check_predict: applying forward pass of the model to get the logits and convert them to probabilities via a softmax");
    let output = resnet34
        .forward_t(&image.unsqueeze(0), /*train=*/ false)
        .softmax(-1, Kind::Float);
    // Finally print the top 5 categories and their associated probabilities.
    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
    //log the prediction results
    log::info!(
        "func: self_check_predict: prediction results: {:?}",
        imagenet::top(&output, 5)
    );
    //returns probability vector with class index
    let top_result = imagenet::top(&output, 1);
    log::info!("Top result: {:?}", top_result);
    let (class, confidence) = top_result.first().unwrap();
    log::info!("Class: {:?}", class);
    log::info!("Confidence: {:?}", confidence);
    let confidence_f64 = match confidence.as_str().parse::<f64>() {
        Ok(value) => value,
        Err(e) => {
            error!("Failed to parse confidence value as f64: {:?}", e);
            0.0 // default value
        }
    };

    info!("func: self_check_predict: prediction result: {:?}", class);
    info!(
        "func: self_check_predict: prediction result: {:?}",
        confidence_f64
    );

    //return
    let prediction = Prediction {
        probabilities: vec![confidence_f64],
        classes: vec![(*class as usize).to_string()],
    };
    //log the result
    info!(
        "func: self_check_predict: prediction result: {:?}",
        prediction
    );
    Ok(prediction)
}

/*
Self check code to verify PyTorch works without models
 */

/*A PyTorch self-tests function */
pub fn tensor_device_cpu() -> String {
    let t = Tensor::of_slice(&[3, 1, 4]);
    let t = t.to_device(Device::Cpu);
    //log descriptive information about the tensor via log::info!
    log::info!("Tensor t: {:?}", t);
    log::info!("Tensor t size: {:?}", t.size());
    log::info!("Tensor t numel: {:?}", t.numel());
    log::info!("Tensor t dim: {:?}", t.dim());
    log::info!("Tensor t kind: {:?}", t.kind());
    log::info!("Tensor t device: {:?}", t.device());
    //return the tensor size as a string
    let size = t.size();
    let size_str = size
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    size_str
}
