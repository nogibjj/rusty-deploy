use actix_multipart::Multipart;
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

pub mod files {
    use std::io::Write;

    use actix_multipart::Multipart;
    use actix_web::{web, Error};
    use futures::{StreamExt, TryStreamExt};
    use log::error;

    pub async fn save_file(mut payload: Multipart, file_path: String) -> Result<String, Error> {
        if let Ok(Some(mut field)) = payload.try_next().await {
            let filepath = format!(".{}", file_path);
            log::info!("func: save_file: filepath: {:?}", filepath);

            let mut f = web::block({
                let filepath_clone = filepath.clone();
                move || std::fs::File::create(&filepath_clone)
            })
            .await
            .map_err(|e| {
                error!("Error creating file: {:?}", e);
                actix_web::error::ErrorInternalServerError("Error creating file")
            })?;

            while let Some(chunk) = field.next().await {
                let data = chunk.unwrap();
                f = web::block({
                    let _filepath_clone = filepath.clone();
                    move || {
                        let mut file = f?;
                        file.write_all(&data)?;
                        Ok(file)
                    }
                })
                .await
                .map_err(|e| {
                    error!("Error writing to file: {:?}", e);
                    actix_web::error::ErrorInternalServerError("Error writing to file")
                })?;
                let metadata = std::fs::metadata(&filepath).unwrap();
                let file_size = metadata.len();
                log::info!("func: save_file: file_size: {:?}", file_size);
            }

            return Ok(filepath);
        }

        Err(actix_web::error::ErrorBadRequest("Error processing file"))
    }
}
/*Self check pre-trained model prediction */
pub fn self_check_predict() -> Result<Prediction, Box<dyn std::error::Error>> {
    log::info!("func: self_check_predict: starting");
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    log::info!("func: self_check_predict: loading image: tests/fixtures/lion.jpg");
    let image = imagenet::load_image_and_resize224("tests/fixtures/lion.jpg").unwrap();
    log::info!("func: self_check_predict: loading model: model/resnet34.ot");
    let weight_file = match env::var("MODEL_PATH") {
        Ok(path) => path,
        Err(_) => "model/resnet34.ot".to_string(),
    };
    let resnet34 = tch::vision::resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT);
    vs.load(weight_file).unwrap();
    log::info!("func: self_check_predict: applying forward pass of the model to get the logits and convert them to probabilities via a softmax");
    let output = resnet34
        .forward_t(&image.unsqueeze(0), /*train=*/ false)
        .softmax(-1, Kind::Float);

    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }

    log::info!(
        "func: self_check_predict: prediction results: {:?}",
        imagenet::top(&output, 5)
    );

    let top_result = imagenet::top(&output, 1);
    log::info!("Top result: {:?}", top_result);
    let (probability, class) = top_result.first().unwrap(); // Swapped variables
    log::info!("Class: {:?}", class);
    log::info!("Confidence: {:?}", probability);
    let confidence_f64 = *probability; // Directly use the probability value

    log::info!("func: self_check_predict: prediction result: {:?}", class);
    log::info!(
        "func: self_check_predict: prediction result: {:?}",
        confidence_f64
    );

    let prediction = Prediction {
        probabilities: vec![confidence_f64],
        classes: vec![class.to_string()], // Updated variable
    };

    log::info!(
        "func: self_check_predict: prediction result: {:?}",
        prediction
    );
    Ok(prediction)
}

pub async fn predict_image(image_path: String) -> Result<Prediction, Box<dyn std::error::Error>> {
    log::info!("route: /predict function: predict_image()");
    log::info!("func: predict_image: loading image: {:?}", image_path);
    let image = imagenet::load_image_and_resize224(&image_path).unwrap();

    log::info!("func: predict_image: starting");
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);

    log::info!("func: predict_image: loading model: model/resnet34.ot");
    let weight_file = match env::var("MODEL_PATH") {
        Ok(path) => path,
        Err(_) => "model/resnet34.ot".to_string(),
    };
    let resnet34 = tch::vision::resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT);
    vs.load(weight_file).unwrap();
    log::info!("func: predict_image:  applying forward pass of the model to get the logits and convert them to probabilities via a softmax");
    let output = resnet34
        .forward_t(&image.unsqueeze(0), /*train=*/ false)
        .softmax(-1, Kind::Float);

    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }

    log::info!(
        "func: predict_image: : prediction results: {:?}",
        imagenet::top(&output, 5)
    );

    let top_result = imagenet::top(&output, 1);
    log::info!("Top result: {:?}", top_result);
    let (probability, class) = top_result.first().unwrap(); // Swapped variables
    log::info!("Class: {:?}", class);
    log::info!("Confidence: {:?}", probability);
    let confidence_f64 = *probability; // Directly use the probability value

    log::info!("func: predict_image: : prediction result: {:?}", class);
    log::info!(
        "func: predict_image: : prediction result: {:?}",
        confidence_f64
    );

    let prediction = Prediction {
        probabilities: vec![confidence_f64],
        classes: vec![class.to_string()], // Updated variable
    };

    log::info!("func: predict_image: : prediction result: {:?}", prediction);
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
