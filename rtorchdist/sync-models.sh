#!/bin/bash

declare -A model_urls=(
  ["resnet18.ot"]="https://github.com/LaurentMazare/tch-rs/releases/download/mw/resnet18.ot"
  ["resnet34.ot"]="https://github.com/LaurentMazare/tch-rs/releases/download/mw/resnet34.ot"
  ["densenet121.ot"]="https://github.com/LaurentMazare/tch-rs/releases/download/mw/densenet121.ot"
  ["vgg13.ot"]="https://github.com/LaurentMazare/tch-rs/releases/download/mw/vgg13.ot"
  ["vgg16.ot"]="https://github.com/LaurentMazare/tch-rs/releases/download/mw/vgg16.ot"
  ["vgg19.ot"]="https://github.com/LaurentMazare/tch-rs/releases/download/mw/vgg19.ot"
  ["squeezenet1_0.ot"]="https://github.com/LaurentMazare/tch-rs/releases/download/mw/squeezenet1_0.ot"
  ["squeezenet1_1.ot"]="https://github.com/LaurentMazare/tch-rs/releases/download/mw/squeezenet1_1.ot"
  ["alexnet.ot"]="https://github.com/LaurentMazare/tch-rs/releases/download/mw/alexnet.ot"
  ["inception-v3.ot"]="https://github.com/LaurentMazare/tch-rs/releases/download/mw/inception-v3.ot"
  ["mobilenet-v2.ot"]="https://github.com/LaurentMazare/tch-rs/releases/download/mw/mobilenet-v2.ot"
)

declare -A model_checksums=(
  ["resnet18.ot"]="your_resnet18_checksum"
  ["resnet34.ot"]="your_resnet34_checksum"
  ["densenet121.ot"]="your_densenet121_checksum"
  ["vgg13.ot"]="your_vgg13_checksum"
  ["vgg16.ot"]="your_vgg16_checksum"
  ["vgg19.ot"]="your_vgg19_checksum"
  ["squeezenet1_0.ot"]="your_squeezenet1_0_checksum"
  ["squeezenet1_1.ot"]="your_squeezenet1_1_checksum"
  ["alexnet.ot"]="your_alexnet_checksum"
  ["inception-v3.ot"]="your_inception_v3_checksum"
  ["mobilenet-v2.ot"]="your_mobilenet_v2_checksum"
)

models_dir="model"

mkdir -p "$models_dir"

for model_name in "${!model_urls[@]}"; do
  url="${model_urls[$model_name]}"
  file_path="$models_dir/$model_name"

  if [ ! -f "$file_path" ]; then
    echo "Downloading $model_name..."
    curl -sSL -o "$file_path" "$url"
  else
    local_checksum=$(sha256sum "$file_path" | awk '{print $1}')
    remote_checksum="${model_checksums[$model_name]}"

    if [ "$local_checksum" != "$remote_checksum" ]; then
      echo "Checksum mismatch for $model_name. Downloading updated file..."
      curl -sSL -o "$file_path" "$url"
    else
      echo "$model_name is up to date. Skipping download."
    fi
  fi
done

echo "All files have been downloaded or updated in the 'model' directory."
