#!/bin/bash

echo "Testing route: /"
curl -X GET http://127.0.0.1:8080/

echo "Testing route: /check_image_prediction"
curl -X GET http://127.0.0.1:8080/check_image_prediction

echo "Testing route: /check_pytorch_cpu"
curl -X GET http://127.0.0.1:8080/check_pytorch_cpu

echo "Testing route: /check_image_upload"
curl -X POST -F "file=@tests/fixtures/lion.jpg" http://127.0.0.1:8080/check_image_upload

echo "Testing route: /predict"
curl -X POST -H "Content-Type: multipart/form-data" -F "image=@tests/fixtures/lion.jpg" http://127.0.0.1:8080/predict

#echo "Testing route: /predict with external image"
#wget "https://upload.wikimedia.org/wikipedia/commons/6/66/An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg" -O cat.jpg
#curl -X POST -H "Content-Type: multipart/form-data" -F "image=@cat.jpg" http://127.0.0.1:8080/predict
#rm cat.jpg

