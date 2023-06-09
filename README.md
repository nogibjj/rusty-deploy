## Rust Deployment Experiments

My hypothesis is RUST is the ideal language for LLMOPs, NOT PYTHON.

### Workflows to Prove Out for Rust MLOps Deployment

#### Distroless

###### Rust T5 Model


##### PyTorch Distroless Pre-Trained Model

![11 10-pytorch-pretrained-models](https://user-images.githubusercontent.com/58792/233207475-c97818ec-5ab2-4e44-947e-dc9296e47ae7.png)



* [Package PyTorch into a Rust Distroless Container](https://github.com/nogibjj/rusty-deploy/tree/main/rtorchdist)
![Screenshot 2023-04-03 at 2 09 38 PM](https://user-images.githubusercontent.com/58792/229592006-9a0c59c1-e1d0-43c7-bc97-dda3d891ec91.png)

##### Hugging Face Distroless Pre-Trained Model

#### Binary Deployment

* Statically Linked ONNX to Rust Binary
* Statically Linked PyTorch to Rust Binary
* Statically Linked HuggingFace to Rust Binary

### References

* [huggingface/text-generation-inference a Rust example](https://github.com/huggingface/text-generation-inference)
