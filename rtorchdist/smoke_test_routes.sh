#!/bin/bash

echo "Testing route: /"
curl -X GET http://127.0.0.1:8080/

echo "Testing route: /check_image_prediction"
curl -X GET http://127.0.0.1:8080/check_image_prediction

echo "Testing route: /check_pytorch_cpu"
curl -X GET http://127.0.0.1:8080/check_pytorch_cpu

echo "Testing route: /check_image_upload"
curl -X POST -F "file=@tests/fixtures/lion.jpg" http://127.0.0.1:8080/check_image_upload